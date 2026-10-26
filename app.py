"""
Smart Bean
------------------
Snap or upload a photo of a specialty coffee bag label, pick your brewing
method and tell the app about your grinder (or pre-ground coffee).

How it works:
1. Gemini reads the label (bean name, origin, process, roast, notes).
2. The app looks up your last 5 logged brews of that bean in coffee_log.csv.
3. Gemini builds a one-cup recipe for your setup, including the grind
   setting in clicks for your grinder, and adjusts it using any past brews
   that have useful tasting notes.
4. You log how it tasted, which improves the next recipe for that bean.

Run:  streamlit run app.py
Needs the GEMINI_API_KEY environment variable to be set.

Steps to run locally:
export GEMINI_API_KEY="{Put your Gemini API key here}"
streamlit run app.py
"""

import difflib
import hashlib
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


MODEL_ID = "gemini-3.6-flash"

BREW_METHODS = [
    "Espresso machine",
    "V60",
    "Hario Switch",
    "AeroPress",
    "Moka pot",
    "French press",
]

# What pre-ground coffee might have been ground for.
PRE_GROUND_OPTIONS = [
    "Espresso",
    "Filter / pour-over",
    "AeroPress",
    "Moka pot",
    "French press",
    "Not sure / general purpose",
]

# How many past brews of the same bean to send to Gemini.
HISTORY_SIZE = 5

LOG_FILE = Path(__file__).parent / "coffee_log.csv"
LOG_COLUMNS = [
    "date",
    "bean_name",
    "roast_level",
    "brew_method",
    "grinder",
    "water_temp_c",
    "grind_size",
    "ratio",
    "tasting_notes",
]


# ---------------------------------------------------------------------------
# What we ask Gemini to return (structured JSON, checked by Pydantic)
# ---------------------------------------------------------------------------

class BeanInfo(BaseModel):
    """Returned by call 1: reading the label."""
    is_coffee_label: bool = Field(
        description="True only if the image shows a coffee bag or coffee label."
    )
    bean_name: str = Field(
        description="Coffee name as printed, e.g. 'Kawa Ethiopia Guji Hambela'. "
        "Start with the roaster if visible, then the coffee name. Use the same "
        "wording every time you see the same label. 'Not listed' if unreadable."
    )
    origin: str = Field(description="Country / region / farm. 'Not listed' if absent.")
    process: str = Field(description="e.g. Washed, Natural, Honey, Anaerobic. 'Not listed' if absent.")
    roast_level: str = Field(
        description="One of: Light, Medium-Light, Medium, Medium-Dark, Dark. "
        "If not printed, infer from cues (e.g. 'filter roast' = Light) and add "
        "'(inferred)' after it."
    )
    tasting_notes: list[str] = Field(description="Tasting notes as printed on the label.")
    other_label_info: str = Field(
        description="Anything else useful for brewing that is printed: variety, "
        "altitude, roast date, the roaster's own recipe. Empty string if none."
    )


class GrindAdvice(BaseModel):
    grinder_identified: str = Field(
        description="Exact grinder make and model you matched the user's text to, "
        "e.g. 'Comandante C40 MK4'. Empty string if the user has no grinder."
    )
    identification_note: str = Field(
        description="How sure you are about the grinder match, and what you assumed "
        "if the text was vague or misspelled. Empty string if no grinder."
    )
    setting: str = Field(
        description="The grind setting on THIS grinder, in its own units. For click "
        "grinders: clicks counted from fully closed, e.g. '24 clicks'. For numbered "
        "dials: e.g. 'Setting 14'. Empty string if no grinder."
    )
    setting_range: str = Field(
        description="A sensible range to experiment within, e.g. '22-26 clicks'. "
        "Empty string if no grinder."
    )
    description: str = Field(
        description="Target grind in plain words with a texture comparison, "
        "e.g. 'Medium-fine, like table salt'."
    )
    suits_method: bool = Field(
        description="True if the user's grinder or pre-ground coffee is suitable for "
        "the chosen brewing method. False if not (e.g. a grinder that cannot go fine "
        "enough for espresso, or French press coffee used in an espresso machine)."
    )
    advice: str = Field(
        description="Short practical advice: how to adjust if the grind doesn't suit "
        "the method, or what grind to ask a café/roaster for. Empty string if none."
    )


class Recipe(BaseModel):
    dose_g: float = Field(description="Coffee in grams for ONE cup.")
    liquid_g: float = Field(
        description="For espresso: the drink yield in the cup, in grams. "
        "For all other methods: total brew water, in grams."
    )
    water_temp_c: int = Field(description="Water temperature in °C.")
    brew_time: str = Field(description="Target total brew time, e.g. '2:45-3:15' or '27-30 s'.")
    steps: list[str] = Field(
        description="Steps in order to brew ONE cup with this method. "
        "Short, one action each, with times and weights."
    )
    why: str = Field(description="One or two short sentences on why this recipe suits this coffee.")


class HistoryReview(BaseModel):
    used: list[str] = Field(
        description="Past brews you used, one line each: '<date>: <what you learned>'. "
        "Empty list if none."
    )
    ignored: list[str] = Field(
        description="Past brews you ignored, one line each: '<date>: <why>'. "
        "Empty list if none."
    )
    changes: str = Field(
        description="In one or two plain sentences, what you changed in this recipe "
        "because of the past brews, e.g. 'Went 2 clicks finer (22) because your last "
        "brew at 24 clicks was sour.' Empty string if nothing changed."
    )


class BrewPlan(BaseModel):
    """Returned by call 2: building the recipe."""
    history: HistoryReview
    grind: GrindAdvice
    recipe: Recipe


LABEL_PROMPT = """
You read specialty coffee bag labels.
- Only report what is printed. Do not invent an origin, process or notes.
- If a field is missing, write "Not listed".
- If the image is not a coffee bag or label, set is_coffee_label to false.
""".strip()

RECIPE_PROMPT = """
You are a specialty coffee barista. You design recipes for ONE cup of
coffee, using the bean details, the user's setup and their past brews.

Step 1 - Learn from past brews (if any are given).
- Each past brew shows the settings the user used and their tasting notes.
- IGNORE a past brew if its notes are not useful for adjusting a recipe:
  empty, vague ("ok", "meh", "fine", "good"), or unrelated to taste
  ("drank it on the train", "new mug"). List it under "ignored" with a reason.
- Also IGNORE a past brew whose bean name is clearly a different coffee
  (e.g. "Kenya AA" vs "Kenya AB"). Small wording differences are fine.
- USE a past brew if its notes say something actionable about taste or
  extraction ("too sour", "bitter finish", "thin body", "perfect, keep it").
  List it under "used" with what you learned.
- Newer brews matter more than older ones.
- Brews with the same method and grinder transfer directly. From a different
  method or grinder, only carry over the general direction (e.g. "this bean
  tends to taste sour"), not the exact setting.
- Typical fixes: sour / thin / weak -> finer grind, hotter water or longer
  brew. Bitter / harsh / dry -> coarser grind, cooler water or shorter brew.
  If a brew was "perfect", keep those settings.
- Say what you changed in "changes". If there are no useful past brews,
  leave "changes" empty and build the standard recipe.

Step 2 - Grind.
- If the user names a grinder, match their text (it may be misspelled,
  partial or a nickname) to the exact make and model. If several models
  fit, pick the most likely one and say so in identification_note.
- Give the setting in that grinder's own units. For hand grinders with
  clicks (Comandante, Timemore, 1Zpresso, Kingrinder, etc.) count clicks
  from fully closed. For numbered dials (Baratza, Fellow Ode, etc.) give
  the number. These are starting points; give a range too.
- If the grinder cannot grind fine enough for the method, set
  suits_method to false and explain in advice.
- If the user has pre-ground coffee, adapt the recipe to that grind
  (e.g. longer or shorter brew time) and warn if it is a poor match.
- If the user has whole beans and no grinder, describe the grind to ask a
  café or roaster for, in advice.

Step 3 - One-cup recipe. Start from these defaults, then adjust for roast
level (lighter = hotter water and a bit finer; darker = cooler and a bit
coarser), for the process and tasting notes, and for the past brews:
- Espresso machine: 18 g in, about 36 g out (1:2; light roasts up to 1:2.5),
  25-30 s, 90-94 °C. liquid_g is the drink yield.
- V60: 15 g coffee, 250 g water, 1:16-1:17, 2:45-3:30 total, 90-96 °C.
- Hario Switch: 15 g coffee, 240 g water, steep with valve closed, then
  release, about 3:00 total, 90-96 °C.
- AeroPress: 15 g coffee, 230 g water, about 2:00 total, 85-95 °C.
- Moka pot: 3-cup pot, about 15 g coffee (fill basket level, don't tamp),
  about 150 g pre-heated water up to the valve, medium-low heat.
- French press: 15 g coffee, 250 g water, coarse, 4:00 steep, 93-96 °C.
If the label includes the roaster's own recipe, take it into account.
""".strip()


# ---------------------------------------------------------------------------
# Gemini calls
# ---------------------------------------------------------------------------

@st.cache_resource
def get_client() -> genai.Client:
    # genai.Client() reads the GEMINI_API_KEY environment variable.
    return genai.Client()


def _ask_gemini(contents: list, system_prompt: str, schema: type[BaseModel],
                thinking_level: str) -> BaseModel:
    response = get_client().models.generate_content(
        model=MODEL_ID,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        ),
    )
    if isinstance(response.parsed, schema):
        return response.parsed
    return schema.model_validate_json(response.text)


# Both calls are cached, so the same input is only sent to Gemini once.
# They return plain dicts because Streamlit's cache can't store Pydantic objects.

@st.cache_data(show_spinner=False)
def _read_label_cached(image_bytes: bytes, mime_type: str) -> dict:
    image = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    # Reading a label is simple, so "low" thinking keeps it fast.
    return _ask_gemini([image, "Read this coffee bag label."],
                       LABEL_PROMPT, BeanInfo, "low").model_dump()


@st.cache_data(show_spinner=False)
def _build_recipe_cached(request_text: str) -> dict:
    # Grinder clicks and learning from past brews need more reasoning,
    # so use the default "medium" level.
    return _ask_gemini([request_text], RECIPE_PROMPT, BrewPlan, "medium").model_dump()


def read_label(image_bytes: bytes, mime_type: str) -> BeanInfo:
    return BeanInfo.model_validate(_read_label_cached(image_bytes, mime_type))


def build_recipe(request_text: str) -> BrewPlan:
    return BrewPlan.model_validate(_build_recipe_cached(request_text))


def describe_setup(method: str, has_grinder: bool, grinder: str,
                   pre_ground: bool | None, ground_for: str | None) -> str:
    """Turn the user's choices into text for Gemini."""
    lines = [f"Brewing method: {method}"]
    if has_grinder:
        lines.append(f"Grinder (user's words): {grinder}")
    elif pre_ground:
        lines.append(f"No grinder. Coffee is pre-ground for: {ground_for}")
    else:
        lines.append("No grinder. Coffee is whole beans (needs grinding at a café or roaster).")
    return "\n".join(lines)


def build_request_text(bean: BeanInfo, setup_text: str, past: pd.DataFrame) -> str:
    """Everything Gemini needs for the recipe, as one block of text."""
    parts = [
        "BEAN (read from the label):",
        bean.model_dump_json(indent=2, exclude={"is_coffee_label"}),
        "",
        "MY SETUP:",
        setup_text,
        "",
    ]
    if past.empty:
        parts.append("MY PAST BREWS OF THIS BEAN: none yet.")
    else:
        parts.append(f"MY PAST BREWS OF THIS BEAN (oldest first, up to {HISTORY_SIZE}):")
        for _, row in past.iterrows():
            parts.append(
                f"- {row['date']} | bean: {row['bean_name']} | "
                f"method: {row['brew_method'] or 'unknown'} | "
                f"grinder: {row['grinder'] or 'unknown'} | grind: {row['grind_size']} | "
                f"water temp: {row['water_temp_c']} °C | ratio: {row['ratio']} | "
                f"notes: \"{row['tasting_notes']}\""
            )
    parts += ["", "Give me a one-cup recipe for this setup."]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Brewing log (CSV)
# ---------------------------------------------------------------------------

def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        # Read everything as text so empty cells become "" instead of NaN.
        return pd.read_csv(LOG_FILE, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log_entry(entry: dict) -> None:
    # Read, add the row, write it all back. This also upgrades older log
    # files that have different columns (old rows just get blanks).
    log = load_log()
    log = pd.concat([log, pd.DataFrame([entry]).astype(str)], ignore_index=True).fillna("")
    columns = LOG_COLUMNS + [c for c in log.columns if c not in LOG_COLUMNS]
    log[columns].to_csv(LOG_FILE, index=False)


def _simplify(name: str) -> str:
    """Lowercase, drop punctuation and extra spaces: 'Kawa – Guji!' -> 'kawa guji'."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(name).lower())).strip()


def find_past_brews(log: pd.DataFrame, bean_name: str, limit: int = HISTORY_SIZE) -> pd.DataFrame:
    """The last `limit` brews of this bean, oldest first.

    Names are matched loosely, because the same label can be read slightly
    differently on different scans ('Kawa Guji' vs 'Kawa - Guji').
    """
    if log.empty or not bean_name or bean_name == "Not listed":
        return log.iloc[0:0]
    target = _simplify(bean_name)
    names = log["bean_name"].map(_simplify)
    close = set(difflib.get_close_matches(target, names.unique(), n=10, cutoff=0.85))
    close.add(target)
    # The log is saved in time order, so the last rows are the newest.
    return log[names.isin(close)].tail(limit)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Smart Bean", page_icon="☕", layout="centered")

st.title("☕ Smart Bean")
st.caption("Scan a coffee bag label → get a one-cup recipe for your setup → "
           "log how it tasted → get a better recipe next time.")

if not os.environ.get("GEMINI_API_KEY"):
    st.error(
        "GEMINI_API_KEY is not set. Get a key at https://aistudio.google.com/apikey, "
        "then run `export GEMINI_API_KEY=\"your-key\"` in the terminal and restart the app."
    )
    st.stop()

# --- 1. Photo ---------------------------------------------------------------
st.subheader("1. Photo of the bag")
source = st.radio("Photo source", ["Upload a photo", "Use camera"], horizontal=True,
                  label_visibility="collapsed")

if source == "Use camera":
    photo = st.camera_input("Point the camera at the label")
else:
    photo = st.file_uploader(
        "Upload a photo of the label",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
    )

if photo is not None and source == "Upload a photo":
    try:
        st.image(photo.getvalue(), width=280)
    except Exception:
        # Some formats (e.g. iPhone HEIC) can't be previewed here,
        # but Gemini can still read them.
        st.caption(f"Photo uploaded ({photo.type}), preview not available.")

# --- 2. Setup ---------------------------------------------------------------
st.subheader("2. Your setup")
method = st.selectbox("Brewing method", BREW_METHODS, index=1)

has_grinder = st.radio("Do you have a grinder?", ["Yes", "No"], horizontal=True) == "Yes"
grinder = ""
pre_ground = None
ground_for = None

if has_grinder:
    grinder = st.text_input(
        "Which grinder do you have?",
        placeholder="e.g. Comandante C40, Timemore C3, 1Zpresso K-Ultra, Baratza Encore",
    ).strip()
else:
    pre_ground = st.radio(
        "Are your coffee beans pre-ground?", ["Yes", "No, whole beans"], horizontal=True
    ) == "Yes"
    if pre_ground:
        ground_for = st.selectbox("What brewing method were they ground for?", PRE_GROUND_OPTIONS)
    else:
        st.caption("No problem. The recipe will tell you what grind to ask a café or roaster for.")

ready = photo is not None and (not has_grinder or grinder != "")
if photo is None:
    st.info("Add a photo of a coffee bag label to get started.")
elif has_grinder and not grinder:
    st.info("Type your grinder's name to continue.")

# The button only sends requests when clicked. Results are remembered
# for this exact photo + setup, so changing a setting asks you to click again.
if ready:
    image_bytes = photo.getvalue()
    mime_type = photo.type or "image/jpeg"
    setup_text = describe_setup(method, has_grinder, grinder, pre_ground, ground_for)
    request_id = hashlib.md5(image_bytes + setup_text.encode()).hexdigest()[:12]

    if st.button("Get my recipe", type="primary"):
        try:
            with st.spinner("Reading the label..."):
                bean = read_label(image_bytes, mime_type)
            plan, past = None, None
            if bean.is_coffee_label:
                past = find_past_brews(load_log(), bean.bean_name)
                with st.spinner("Building your recipe..."):
                    plan = build_recipe(build_request_text(bean, setup_text, past))
            st.session_state["result"] = {
                "id": request_id, "bean": bean, "plan": plan, "past": past,
            }
        except Exception as err:  # network issues, bad key, quota, etc.
            st.error(f"Gemini request failed: {err}")

    result = st.session_state.get("result")
    if result and result["id"] != request_id:
        st.caption("Your photo or setup changed. Click **Get my recipe** to update it.")

    # --- 3. Results ---------------------------------------------------------
    if result and result["id"] == request_id:
        bean: BeanInfo = result["bean"]

        if not bean.is_coffee_label:
            st.warning("That doesn't look like a coffee label. Try a clearer photo of the bag.")
        else:
            plan: BrewPlan = result["plan"]
            past: pd.DataFrame = result["past"]
            grind, recipe, history = plan.grind, plan.recipe, plan.history

            st.subheader("3. Bean details")
            st.markdown(f"**{bean.bean_name}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Origin", bean.origin)
            c2.metric("Process", bean.process)
            c3.metric("Roast", bean.roast_level)
            if bean.tasting_notes:
                st.markdown("**Tasting notes:** " + ", ".join(bean.tasting_notes))

            # --- What Gemini learned from past brews ----------------------
            st.subheader("4. Learning from your past brews")
            if past is None or past.empty:
                st.caption("No past brews of this bean yet. Log this brew below and "
                           "the next recipe will take it into account.")
            else:
                if history.changes:
                    st.success(history.changes)
                else:
                    st.caption("None of your past brews had notes that called for a change, "
                               "so this is the standard recipe.")
                with st.expander(f"Past brews checked ({len(past)})"):
                    for line in history.used:
                        st.markdown(f"✅ **Used:** {line}")
                    for line in history.ignored:
                        st.markdown(f"➖ **Ignored:** {line}")

            # --- Grind ----------------------------------------------------
            st.subheader("5. Grind")
            if has_grinder:
                st.markdown(f"**Grinder:** {grind.grinder_identified}")
                if grind.identification_note:
                    st.caption(grind.identification_note)
                g1, g2 = st.columns(2)
                g1.metric("Start at", grind.setting)
                g2.metric("Range to try", grind.setting_range)
            st.markdown(f"**Target grind:** {grind.description}")
            if not grind.suits_method:
                st.warning(grind.advice or "This grind isn't a good match for this method.")
            elif grind.advice:
                st.info(grind.advice)
            if has_grinder:
                st.caption("Click settings are a starting point. Go finer if it tastes sour, "
                           "coarser if it tastes bitter.")

            # --- Recipe ---------------------------------------------------
            st.subheader(f"6. {method} recipe (1 cup)")
            is_espresso = method == "Espresso machine"
            ratio = f"1:{recipe.liquid_g / recipe.dose_g:.1f}"
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Coffee", f"{recipe.dose_g:g} g")
            m2.metric("Yield" if is_espresso else "Water", f"{recipe.liquid_g:g} g")
            m3.metric("Temp", f"{recipe.water_temp_c} °C")
            m4.metric("Ratio", ratio)
            st.markdown(f"**Brew time:** {recipe.brew_time}")
            st.markdown("\n".join(f"{i}. {step}" for i, step in enumerate(recipe.steps, start=1)))
            st.caption(recipe.why)

            # --- Brewing log form -----------------------------------------
            st.subheader("7. Brewing log")
            if has_grinder:
                default_grind = f"{grind.setting} ({grind.grinder_identified})"
                grinder_label = grind.grinder_identified
            else:
                default_grind = f"Pre-ground for {ground_for}" if pre_ground else grind.description
                grinder_label = "None"

            # Form key includes the request id so the defaults refresh for each recipe.
            with st.form(f"log_form_{request_id}", clear_on_submit=True):
                bean_name = st.text_input(
                    "Bean", value=bean.bean_name,
                    help="Keep this name the same for every brew of this bag, so the "
                         "app can find your past brews.",
                )
                f1, f2 = st.columns(2)
                temp_used = f1.number_input(
                    "Actual water temp used (°C)", min_value=70, max_value=100,
                    value=min(max(recipe.water_temp_c, 70), 100),
                )
                ratio_used = f2.text_input("Coffee-to-water ratio", value=ratio)
                grind_used = st.text_input("Grind size used", value=default_grind)
                notes = st.text_area(
                    "Tasting notes",
                    placeholder="e.g. Too sour, need finer grind. (Specific notes "
                                "help the next recipe; 'ok' gets ignored.)",
                )

                if st.form_submit_button("Save brew"):
                    save_log_entry(
                        {
                            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "bean_name": bean_name.strip(),
                            "roast_level": bean.roast_level,
                            "brew_method": method,
                            "grinder": grinder_label,
                            "water_temp_c": temp_used,
                            "grind_size": grind_used.strip(),
                            "ratio": ratio_used.strip(),
                            "tasting_notes": notes.strip(),
                        }
                    )
                    st.success("Brew saved. Click **Get my recipe** again any time to "
                               "get a recipe that uses this brew.")

# --- Past brews -------------------------------------------------------------
st.divider()
st.subheader("Past brews")
log = load_log()
if log.empty:
    st.caption("No brews logged yet.")
else:
    # Newest first
    st.dataframe(log.iloc[::-1], hide_index=True)
