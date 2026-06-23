"""File-backed CLAP vocabulary lists (parallel mood/style/imagery channels).

Each list lives in its own JSON under ``clap_vocabs/``. Structural fields
(``phrases``, ``id``, ``name``) are edited in the file; runtime tuning
(``enable``, thresholds, ``top_k``, etc.) is exposed in the UI and saved back
to the same file on Save Defaults.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from music_visualizer.config import Config
from music_visualizer.utils.logging import get_logger
from music_visualizer.utils.paths import data_dir

logger = get_logger(__name__)

VOCAB_KEY_PREFIX = "vocab"
VOCAB_UI_FIELDS = frozenset({
    "enable", "interval_sec", "top_k", "max_tags", "enter_threshold", "exit_threshold",
    "score_tau", "min_hold_sec", "weight_min", "weight_max", "weight_step",
    "softmax_temp", "calibrate", "calib_alpha", "calib_warmup", "calib_spike_guard",
})

DEFAULT_VOCAB_LIST_PARAMS: Dict[str, Any] = {
    "enable": True,
    "interval_sec": 1.0,
    "top_k": 5,
    "max_tags": 2,
    "enter_threshold": 0.12,
    "exit_threshold": 0.06,
    "score_tau": 1.5,
    "min_hold_sec": 1.5,
    "weight_min": 0.7,
    "weight_max": 1.4,
    "weight_step": 0.1,
    "softmax_temp": 0.02,
    "calibrate": True,
    "calib_alpha": 0.001,
    "calib_warmup": 8,
    "calib_spike_guard": 0.05,
}

_FILE_COMMENT = (
    "CLAP vocabulary list. Edit phrases here; tune runtime fields in the UI "
    "(saved back to this file on Save Defaults). See docs/clap_vocab_lists.md."
)


@dataclass
class VocabList:
    """One CLAP scoring channel backed by a JSON file."""

    id: str
    name: str
    phrases: List[str]
    order: int = 0
    group: str = ""
    enable: bool = True
    interval_sec: float = 1.0
    top_k: int = 5
    max_tags: int = 2
    enter_threshold: float = 0.12
    exit_threshold: float = 0.06
    score_tau: float = 1.5
    min_hold_sec: float = 1.5
    weight_min: float = 0.7
    weight_max: float = 1.4
    weight_step: float = 0.1
    softmax_temp: float = 0.02
    calibrate: bool = True
    calib_alpha: float = 0.001
    calib_warmup: int = 8
    calib_spike_guard: float = 0.05
    _source_path: str = field(default="", repr=False)

    def display_group(self) -> str:
        return self.group or f"CLAP: {self.name}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_source_path", None)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, source_path: str = "") -> "VocabList":
        phrases_raw = data.get("phrases", [])
        if not isinstance(phrases_raw, list):
            phrases_raw = []
        phrases = [str(x).strip() for x in phrases_raw if str(x).strip()]
        vid = str(data.get("id", "")).strip()
        name = str(data.get("name", vid or "Vocab")).strip()
        params = {**DEFAULT_VOCAB_LIST_PARAMS}
        for key in DEFAULT_VOCAB_LIST_PARAMS:
            if key in data:
                params[key] = data[key]
        return cls(
            id=vid or _slugify(name),
            name=name,
            phrases=phrases,
            order=int(data.get("order", 0)),
            group=str(data.get("group", "")),
            enable=bool(params["enable"]),
            interval_sec=float(params["interval_sec"]),
            top_k=int(params["top_k"]),
            max_tags=int(params["max_tags"]),
            enter_threshold=float(params["enter_threshold"]),
            exit_threshold=float(params["exit_threshold"]),
            score_tau=float(params["score_tau"]),
            min_hold_sec=float(params["min_hold_sec"]),
            weight_min=float(params["weight_min"]),
            weight_max=float(params["weight_max"]),
            weight_step=float(params["weight_step"]),
            softmax_temp=float(params["softmax_temp"]),
            calibrate=bool(params["calibrate"]),
            calib_alpha=float(params["calib_alpha"]),
            calib_warmup=int(params["calib_warmup"]),
            calib_spike_guard=float(params["calib_spike_guard"]),
            _source_path=source_path,
        )


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "vocab"


def vocabs_dir_path(cfg: Config) -> Path:
    if cfg.clap_vocabs_dir:
        return Path(cfg.clap_vocabs_dir)
    return data_dir() / "clap_vocabs"


def normalize_vocab_lists(lists: Optional[List[dict]]) -> List[dict]:
    if not lists:
        return []
    out: List[dict] = []
    for item in lists:
        vl = VocabList.from_dict(item, source_path=str(item.get("_source_path", "")))
        d = vl.to_dict()
        if vl._source_path:
            d["_source_path"] = vl._source_path
        out.append(d)
    out.sort(key=lambda x: (int(x.get("order", 0)), str(x.get("id", ""))))
    return out


def load_vocab_lists_dir(path: Path) -> Optional[List[dict]]:
    """Load all ``*.json`` vocab files from ``path``; None if dir missing/empty."""
    if not path.is_dir():
        return None
    loaded: List[dict] = []
    for fp in sorted(path.glob("*.json")):
        try:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping vocab file %s (%s).", fp, exc)
            continue
        if not isinstance(data, dict):
            logger.warning("Vocab file %s must be a JSON object; skipping.", fp)
            continue
        vl = VocabList.from_dict(data, source_path=str(fp))
        if not vl.phrases:
            logger.warning("Vocab file %s has no phrases; skipping.", fp)
            continue
        if not vl.id:
            logger.warning("Vocab file %s has no id; skipping.", fp)
            continue
        d = vl.to_dict()
        d["_source_path"] = str(fp)
        loaded.append(d)
    if not loaded:
        return None
    return normalize_vocab_lists(loaded)


def save_vocab_list_file(path: Path, data: dict[str, Any]) -> Path:
    """Write one vocab list JSON, preserving phrases and runtime fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vl = VocabList.from_dict(data, source_path=str(path))
    payload = vl.to_dict()
    payload["_comment"] = _FILE_COMMENT
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return path


def save_all_vocab_lists(cfg: Config) -> List[Path]:
    """Persist every loaded vocab list back to its source file."""
    paths: List[Path] = []
    for item in normalize_vocab_lists(cfg.clap_vocab_lists):
        src = item.get("_source_path")
        if not src:
            continue
        p = save_vocab_list_file(Path(src), item)
        paths.append(p)
    return paths


def parse_vocab_key(key: str) -> Optional[tuple[int, str]]:
    """Parse ``vocab2_top_k`` -> ``(2, 'top_k')``."""
    if not key.startswith(VOCAB_KEY_PREFIX) or len(key) < len(VOCAB_KEY_PREFIX) + 2:
        return None
    rest = key[len(VOCAB_KEY_PREFIX):]
    idx_str = ""
    for ch in rest:
        if ch.isdigit():
            idx_str += ch
        else:
            break
    if not idx_str or not rest[len(idx_str):].startswith("_"):
        return None
    field_name = rest[len(idx_str) + 1:]
    if field_name not in VOCAB_UI_FIELDS:
        return None
    return int(idx_str), field_name


def vocab_field_value(cfg: Config, index: int, field_name: str) -> Any:
    lists = normalize_vocab_lists(cfg.clap_vocab_lists)
    if index >= len(lists):
        return None
    vl = VocabList.from_dict(lists[index])
    return getattr(vl, field_name)


def set_vocab_field(cfg: Config, index: int, field_name: str, value: Any) -> None:
    lists = normalize_vocab_lists(cfg.clap_vocab_lists)
    if index >= len(lists):
        return
    vl = VocabList.from_dict(lists[index])
    if field_name == "enable" or field_name == "calibrate":
        value = bool(value)
    elif field_name in ("top_k", "max_tags", "calib_warmup"):
        value = int(value)
    else:
        value = float(value)
    setattr(vl, field_name, value)
    d = vl.to_dict()
    src = lists[index].get("_source_path")
    if src:
        d["_source_path"] = src
    lists[index] = d
    cfg.clap_vocab_lists = lists


def build_default_vocab_seed() -> List[dict]:
    """Return the three built-in default vocab list dicts (no paths yet)."""
    seeds = [
        ("mood", "Mood", 0, "CLAP: Mood", default_mood_phrases()),
        ("style", "Style", 1, "CLAP: Style", default_style_phrases()),
        ("imagery", "Imagery", 2, "CLAP: Imagery", default_imagery_phrases()),
    ]
    out: List[dict] = []
    for vid, name, order, group, phrases in seeds:
        d = {
            "id": vid,
            "name": name,
            "order": order,
            "group": group,
            "phrases": phrases,
            **DEFAULT_VOCAB_LIST_PARAMS,
        }
        out.append(d)
    return out


def seed_default_vocab_files(path: Path) -> List[dict]:
    """Create default mood/style/imagery JSON files and return loaded list."""
    path.mkdir(parents=True, exist_ok=True)
    loaded: List[dict] = []
    for item in build_default_vocab_seed():
        fp = path / f"{item['id']}.json"
        save_vocab_list_file(fp, item)
        d = dict(item)
        d["_source_path"] = str(fp)
        loaded.append(d)
    logger.info("Created default CLAP vocab lists in %s", path)
    return normalize_vocab_lists(loaded)


def default_mood_phrases() -> List[str]:
    return [
        "wistful nostalgia", "bittersweet resignation", "feral euphoria",
        "somber reverence", "anxious anticipation", "defiant triumph",
        "lonely midnight calm", "restless wanderlust", "grief-stricken stillness",
        "playful mischief", "hypnotic trance", "volatile rage",
        "tender vulnerability", "existential dread", "blissful surrender",
        "melancholic longing", "triumphant catharsis", "suffocating tension",
        "ethereal detachment", "reckless abandon", "quiet devastation",
        "feverish urgency", "serene acceptance", "bitter contempt",
        "innocent wonder", "paranoid vigilance", "sensual languor",
        "stoic endurance", "manic exhilaration", "forlorn isolation",
        "reverent awe", "petulant defiance", "wistful yearning",
        "crushing despair", "delirious joy", "brooding introspection",
        "fierce determination", "haunted regret", "giddy elation",
        "numb dissociation", "sacred solemnity", "impish delight",
        "smoldering resentment", "fragile hope", "wild ecstasy",
        "solemn gravitas", "skittish unease", "languid contentment",
        "vengeful fury", "wistful daydream", "solemn mourning",
        "electric anticipation", "weary resignation", "radiant optimism",
        "clenched frustration", "dreamy reverie", "stark alienation",
        "jubilant celebration", "cold indifference", "warm nostalgia",
        "panicked urgency", "gentle melancholy", "fierce pride",
        "listless boredom", "intoxicated bliss", "solemn penitence",
        "spiteful malice", "hushed intimacy", "stormy turbulence",
        "childlike innocence", "world-weary cynicism", "transcendent rapture",
        "aching sorrow", "defiant resilience", "seductive allure",
        "grim fatalism", "sparkling mischief", "solemn tribute",
        "restless impatience", "deep contemplation", "frenzied chaos",
        "peaceful resignation", "smoldering passion", "icy detachment",
        "tender affection", "raw anguish", "quiet triumph",
        "delicate wistfulness", "unbridled enthusiasm", "solemn dread",
        "cozy warmth", "sharp betrayal", "floating serenity",
        "crushing guilt", "bright exuberance", "shadowed foreboding",
        "hollow emptiness", "fierce loyalty", "drunken euphoria",
        "quiet defiance", "aching tenderness", "wild desperation",
        "calm before storm", "giddy anticipation", "solemn farewell",
        "bittersweet closure", "feverish obsession", "gentle forgiveness",
        "smoldering jealousy", "radiant gratitude", "numb shock",
        "fierce protectiveness", "languid seduction", "stark vulnerability",
        "reckless bravado", "quiet shame", "explosive release",
        "wistful remembrance", "cold resolve", "warm belonging",
        "paralyzing fear", "quiet courage", "delirious abandon",
        "solemn initiation", "petty irritation", "deep reverence",
        "restless hunger", "gentle resignation", "fierce ecstasy",
        "brooding menace", "innocent trust", "world-shattering grief",
        "quiet elation", "smoldering desire", "stark clarity",
        "frenzied liberation", "hushed wonder", "bitter loneliness",
        "radiant confidence", "crushing inevitability", "playful defiance",
        "solemn awakening", "aching nostalgia", "wild tenderness",
        "numb routine", "fierce mourning", "gentle euphoria",
        "restless nostalgia", "quiet fury", "blinding hope",
        "suffocating love", "ethereal sadness", "grounded contentment",
        "volatile tenderness", "solemn ecstasy", "wistful surrender",
        "feverish nostalgia", "quiet rebellion", "deep contentment",
        "smoldering grief", "radiant defiance", "hollow longing",
        "fierce serenity", "gentle dread", "explosive tenderness",
        "brooding euphoria", "innocent grief", "world-weary hope",
        "quiet obsession", "delirious tenderness", "solemn rage",
        "restless serenity", "bittersweet triumph", "numb euphoria",
        "fierce nostalgia", "gentle fury", "wild serenity",
        "haunted euphoria", "tender dread", "volatile serenity",
        "wistful fury", "solemn tenderness", "radiant grief",
        "quiet ecstasy", "smoldering hope", "fragile defiance",
        "crushing tenderness", "delirious grief", "ethereal fury",
        "grounded euphoria", "feverish serenity", "hushed defiance",
        "bittersweet fury", "wild nostalgia", "solemn obsession",
        "aching defiance", "gentle obsession", "fierce wistfulness",
        "numb defiance", "radiant obsession", "quiet wistfulness",
        "smoldering wistfulness", "volatile obsession", "brooding defiance",
        "innocent obsession", "world-weary defiance", "transcendent wistfulness",
    ]


def default_style_phrases() -> List[str]:
    return [
        "chromatic aberration glitch art", "bioluminescent ink wash",
        "retro CRT phosphor bloom", "baroque chiaroscuro oil painting",
        "isometric pixel art diorama", "infrared false-color photography",
        "art nouveau stained glass", "brutalist concrete relief",
        "surrealist double exposure", "ukiyo-e woodblock with gold leaf",
        "vaporwave pastel gradient grid", "hand-tinted daguerreotype",
        "low-poly faceted 3D render", "infrared heat-map palette",
        "cross-processed film grain", "neon-noir cyberpunk illustration",
        "impasto palette-knife texture", "cyanotype blueprint aesthetic",
        "tilt-shift miniature effect", "rotoscoped animation frame",
        "risograph misregistration print", "holographic foil shimmer",
        "wet-plate collodion portrait", "ASCII art monochrome",
        "anamorphic lens flare streak", "fresco secco wall painting",
        "pointillist stipple dots", "inverted negative silhouette",
        "cel-shaded toon outline", "long-exposure light painting",
        "polaroid instant fade border", "embossed metallic engraving",
        "mosaic tesserae tile", "scratchboard white-on-black",
        "halftone newspaper dot screen", "kinetic op-art stripes",
        "infrared foliage pink shift", "thermal imaging false color",
        "wireframe vector overlay", "claymation stop-motion texture",
        "encaustic wax layered medium", "gouache matte flat color",
        "pastel chalk soft edge", "charcoal smudge sketch",
        "silver gelatin print contrast", "lomography vignette blur",
        "double-diamond isometric tile", "voxel art block sculpture",
        "paper-cut silhouette layer", "stained-glass lead came lines",
        "fresco buon wet plaster", "tempera egg binder sheen",
        "encaustic hot wax drip", "sgraffito scratched reveal",
        "fresco trompe-l'oeil depth", "grisaille monochrome underpainting",
        "verdaccio green underpainting", "sfumato soft smoke blend",
        "tenebrism dramatic shadow", "chiaroscuro single light source",
        "impasto thick paint ridges", "alla prima wet-on-wet",
        "glazing transparent layers", "scumbling dry brush haze",
        "pointillism pure color dots", "divisionism optical mix",
        "fauvism wild non-local color", "expressionist distorted form",
        "cubist fractured plane", "futurist motion blur lines",
        "dadaist collage cutout", "constructivist geometric bold",
        "suprematist pure shape", "de stijl primary grid",
        "bauhaus functional geometry", "art deco geometric luxury",
        "streamline moderne curve", "mid-century modern flat",
        "pop art ben-day dots", "op art vibrating pattern",
        "minimalist reductive form", "photorealist hyper detail",
        "hyperreal glossy surface", "magic realism subtle oddity",
        "symbolist allegorical mood", "pre-raphaelite jewel tone",
        "academic salon polish", "romantic sublime landscape",
        "hudson river luminous light", "impressionist broken brushstroke",
        "post-impressionist outline", "neo-impressionist dot matrix",
        "expressionist woodcut gouge", "woodblock moku-hanga water",
        "linocut bold contrast cut", "etching aquatint grain",
        "mezzotint velvet black", "drypoint burr fuzz line",
        "lithograph crayon texture", "screenprint flat spot color",
        "monotype ghost second pull", "collagraph cardboard plate",
        "photogravure continuous tone", "carbon print rich black",
        "albumen print sepia tone", "cyanotype prussian blue",
        "van dyke brown print", "platinum palladium matte",
        "salt print warm highlight", "ambrotype glass positive",
        "tintype metal plate portrait", "stereo card dual view",
        "stereoscopic anaglyph red-cyan", "autochrome potato starch grain",
        "kodachrome saturated slide", "ektar vivid punch color",
        "portra negative soft skin", "cross-process green-magenta shift",
        "bleach bypass silver retain", "push-process grain boost",
        "pull-process muted contrast", "infrared black sky white foliage",
        "schlieren flow visualization", "kirlian corona discharge aura",
        "electron microscope false color", "SEM grayscale micro texture",
        "x-ray radiograph bone ghost", "MRI slice false rainbow",
        "sonogram speckle grain", "lidar height color ramp",
        "satellite multispectral band", "NDVI vegetation red edge",
        "astrophotography star trail", "deep-sky narrowband palette",
        "solar telescope hydrogen alpha", "planetary stacked detail",
        "microscopy darkfield glow", "phase contrast halo edge",
        "fluorescence UV excite glow", "polarized light crystal color",
        "schlieren shock wave ripple", "high-speed flash freeze frame",
        "bullet-time frozen array", "motion vector field arrow",
        "optical flow color warp", "datamosh compression artifact",
        "pixel sorting streak glitch", "databending raw corruption",
        "feedback loop video smear", "analog TV roll tear",
        "VHS tracking noise band", "betamax color bleed",
        "gameboy four-green palette", "CGA four-color limit",
        "EGA sixteen-color dither", "VGA 256-color gradient",
        "demoscene plasma effect", "raymarched signed distance",
        "path-traced caustic glass", "volumetric god-ray fog",
        "subsurface skin scatter", "iridescent thin-film oil",
        "anisotropic brushed metal", "procedural marble vein",
        "fractal mandelbrot zoom", "reaction-diffusion Turing pattern",
        "cellular automata life grid", "voronoi stained cell edge",
        "delaunay triangulation mesh", "quasicrystal penrose tile",
        "isometric axonometric no vanish", "oblique cavalier projection",
        "fish-eye barrel distortion", "panini compressed perspective",
        "little planet stereographic", "equirectangular 360 unwrap",
        "cubemap six-face skybox", "matcap studio ball shade",
        "toon ramp stepped shade", "hatching cross contour line",
        "engraving parallel hatch", "wood engraving white line",
        "lithograph tusche wash tone", "monotype ghost impression",
        "collage magazine tear edge", "decollage ripped layer reveal",
        "assemblage found object shadow", "bricolage mixed media grit",
        "frottage rub texture transfer", "grattage scrape reveal under",
        "automatism drip pour flow", "action painting splatter arc",
        "color field flat expanse", "hard-edge acrylic crisp",
        "shaped canvas irregular outline", "relief sculpture cast shadow",
        "bas-relief shallow carve", "high-relief deep projection",
        "lost-wax bronze patina", "verdigris copper oxidize green",
        "repoussé hammered metal form", "chasing fine surface detail",
        "filigree wire lace metal", "cloisonné enamel cell fill",
        "champlevé recessed enamel", "niello silver sulfur inlay",
        "damascene gold inlay steel", "marquetry wood veneer puzzle",
        "parquetry geometric wood floor", "intarsia pictorial wood inlay",
        "stained concrete acid wash", "terrazzo chip aggregate floor",
        "mosaic smalti glass tile", "pebble mosaic river stone",
        "kintsugi gold crack repair", "wabi-sabi imperfect glaze crack",
        "raku fire-blackened ceramic", "celadon crackle glaze web",
        "tenmoku oil-spot iron glaze", "shino white feldspar fleck",
        "sgraffito scratched slip line", "majolica tin white opaque",
        "faience turquoise egyptian", "porcelain blue underglaze",
        "bone china translucent thin", "stoneware salt glaze orange peel",
    ]


def default_imagery_phrases() -> List[str]:
    return [
        "horseback rider at dusk", "sliced lemons on ice", "UFO over wheat field",
        "empty beach at sunrise", "broken mirror shards", "koi pond with lily pads",
        "neon vending machine alley", "hot air balloons at dawn", "rusted anchor on pier",
        "moth circling bare bulb", "windmill against orange sky", "cobblestone rain puddle",
        "red telephone booth fog", "desert highway mirage", "snow-covered pine forest",
        "volcano erupting at night", "lighthouse beam through mist", "carousel horses spinning",
        "train crossing steel bridge", "butterfly on wilted rose", "typewriter with blank page",
        "grand piano in empty hall", "stack of old vinyl records", "vinyl record spinning",
        "cathedral stained glass sunbeam", "market stall overflowing fruit", "origami crane flock",
        "campfire sparks into stars", "tent under milky way", "canoe on still lake",
        "waterfall rainbow mist", "coral reef tropical fish", "jellyfish drifting deep blue",
        "octopus tentacle close-up", "whale breaching ocean spray", "dolphin arc jump",
        "seahorse in kelp forest", "starfish on tide pool rock", "crab scuttling sand",
        "seagull over crashing wave", "sailboat white canvas horizon", "submarine porthole view",
        "deep sea anglerfish lure", "shipwreck coral encrusted", "treasure chest gold coins",
        "pirate flag skull crossbones", "compass rose brass antique", "map with X marks spot",
        "telescope pointed at moon", "observatory dome open night", "satellite dish array desert",
        "rocket launch pad smoke", "space shuttle exhaust plume", "astronaut floating tether",
        "moon footprint gray dust", "mars rover red dunes", "asteroid belt rocky drift",
        "comet tail streak sky", "meteor shower streak lines", "eclipse corona ring",
        "northern lights green curtain", "thunderstorm lightning fork", "tornado funnel dark sky",
        "rainbow over farmland", "hail stones on pavement", "fog rolling over hills",
        "dust devil desert spin", "sandstorm orange wall", "blizzard whiteout street",
        "icicle hanging eave", "frost pattern window glass", "snowman carrot nose",
        "sled downhill children", "ice skaters frozen pond", "aurora reflected lake",
        "campfire marshmallow roast", "lantern festival sky release", "fireworks burst night sky",
        "sparkler trail dark", "bonfire beach night", "candlelit dinner table",
        "chandelier crystal ballroom", "streetlamp wet pavement glow", "neon sign buzzing hum",
        "billboard highway dusk", "traffic light red glow rain", "crosswalk zebra stripes",
        "subway tunnel arriving train", "escalator descending blur", "airport runway lights",
        "control tower glass cab", "hang glider cliff launch", "paraglider mountain ridge",
        "skydiver freefall spread", "biplane barnstorm loop", "helicopter searchlight night",
        "drone overhead grid pattern", "satellite orbit earth curve", "space station solar panel",
        "moon base dome habitat", "mars colony red dust", "alien artifact glowing obelisk",
        "crop circle field pattern", "stonehenge solstice sunrise", "pyramid desert silhouette",
        "sphinx sand wind erosion", "colosseum ruin ivy", "eiffel tower night sparkle",
        "big ben clock fog", "statue of liberty torch", "golden gate bridge fog",
        "mount fuji cherry blossom", "great wall snaking ridge", "taj mahal reflection pool",
        "angkor wat temple vine", "machu picchu cloud forest", "petra rose city canyon",
        "santorini white blue dome", "venice gondola canal", "amsterdam canal bicycles",
        "london double decker bus", "yellow taxi rain nyc", "tokyo crossing crowd blur",
        "shibuya scramble neon", "times square billboard glow", "las vegas strip night",
        "hollywood sign hillside", "route 66 diner neon", "gas station desert night",
        "motel vacancy sign flicker", "drive-in movie screen", "roller rink disco ball",
        "bowling alley pin strike", "arcade cabinet glow dark", "pinball machine lights",
        "jukebox chrome diner", "soda fountain milkshake", "diner booth red vinyl",
        "kitchen copper pots hanging", "cast iron skillet sizzle", "bread oven fire glow",
        "farmers market tomato crate", "honey jar golden drip", "cheese wheel cellar age",
        "wine barrel cellar dim", "champagne cork pop", "espresso machine steam",
        "coffee cup morning steam", "tea ceremony matcha bowl", "sake bottle tokkuri",
        "whiskey glass ice clink", "cocktail shaker pour", "bar counter neon back",
        "pool table green felt", "dartboard bullseye hit", "chess board mid-game",
        "poker chips all-in push", "slot machine jackpot flash", "roulette wheel spin blur",
        "playing card ace reveal", "dice roll snake eyes", "domino chain falling",
        "marble run spiral drop", "yo-yo string trick loop", "kite tail windy beach",
        "paper airplane glide", "model rocket launch park", "train set loop village",
        "dollhouse miniature room", "lego castle brick build", "jigsaw puzzle last piece",
        "rubiks cube solved side", "spinning top wobble stop", "music box ballerina turn",
        "metronome ticking swing", "gramophone horn spin", "microphone stage spotlight",
        "guitar pick strum blur", "violin bow rosined stroke", "drumstick snare hit",
        "saxophone jazz club smoke", "trumpet mute wah-wah", "accordion street performer",
        "harmonica blues train", "banjo porch sunset", "harp angelic glow",
        "organ pipe cathedral echo", "choir robes candle aisle", "conductor baton raise",
        "ballet tutu spin blur", "tap shoes spark floor", "breakdance freeze pose",
        "circus tightrope balance", "clown red nose balloon", "magician rabbit hat",
        "tarot card spread table", "crystal ball fog swirl", "ouija board candle dark",
        "hourglass sand falling", "pocket watch chain vest", "alarm clock ringing morning",
        "calendar page tear away", "postcard stamp corner", "love letter wax seal",
        "newspaper headline fold", "book open candle read", "library ladder tall shelf",
        "scroll ancient parchment", "quill ink blot page", "printing press type block",
        "camera flash bulb pop", "polaroid photo develop", "film reel projector beam",
        "television static snow", "radio dial glow dial", "telephone rotary dial",
        "computer CRT green text", "floppy disk label hand", "cassette tape unspool",
        "walkman headphones cord", "cd rainbow reflection", "smartphone screen glow face",
        "laptop coffee shop typing", "server rack blinking led", "robot arm assembly line",
        "android humanoid face", "cyborg eye red glow", "mech suit battle damage",
        "tank tread mud track", "fighter jet afterburner", "aircraft carrier deck crew",
        "battleship gray ocean", "submarine periscope rise", "torpedo wake trail",
        "missile contrail sky", "bunker concrete blast door", "barbed wire fence fog",
        "watchtower searchlight sweep", "helmet visor reflection war", "medal ribbon chest",
        "flag waving wind pole", "peace dove olive branch", "olive branch white dove",
        "olive tree ancient grove", "vineyard grape harvest basket", "wheat field golden sway",
        "corn maze aerial view", "sunflower field facing sun", "pumpkin patch autumn",
        "apple orchard ladder pick", "cherry blossom petal fall", "maple leaf red stream",
        "mushroom forest fairy ring", "fern grotto waterfall mist", "bamboo grove wind sway",
        "cactus desert bloom pink", "succulent terrarium glass", "bonsai twisted trunk",
        "topiary hedge animal shape", "greenhouse steam tropical", "garden gnome flower bed",
        "bird bath sparrow splash", "bird nest eggs blue", "eagle soaring mountain peak",
        "owl perched moon branch", "raven on gothic spire", "peacock tail fan display",
        "flamingo pink lagoon stand", "penguin iceberg slide", "polar bear ice floe",
        "wolf howl full moon", "deer forest mist antlers", "fox snow trail print",
        "rabbit meadow hop blur", "squirrel acorn oak tree", "hedgehog garden path",
        "turtle sunning log pond", "frog lily pad croak", "snake coiled rock sun",
        "lizard desert sand dash", "crocodile river eyes surface", "shark fin ocean cut",
        "horse gallop dust trail", "bull rodeo dust kick", "cow pasture bell neck",
        "sheep wool cloud field", "goat mountain cliff stand", "pig mud wallow pink",
        "chicken coop morning crow", "rooster barn sunrise", "duck pond paddle ripple",
        "swan lake graceful curve", "goose V formation migrate", "pigeon city statue perch",
        "parrot tropical perch bright", "canary cage yellow song", "hummingbird flower hover",
        "bee hive honeycomb hex", "butterfly garden monarch", "dragonfly pond iridescent",
        "firefly jar summer night", "ladybug red spot leaf", "spider web dew drop",
        "scorpion desert uv glow", "ant colony trail carry", "beetle iridescent shell",
        "caterpillar leaf munch", "chrysalis hanging branch", "snail shell garden slow",
        "slug trail silver slime", "earthworm soil rich dark", "fish aquarium bubble rise",
        "goldfish bowl round swim", "betta fish fin flare", "koi orange white pond",
        "lobster trap buoy float", "oyster pearl inside shell", "clam sand beach open",
        "mussel rock tide line", "squid ink cloud escape", "nautilus shell spiral math",
        "shell sand dollar beach", "coral pink branch reef", "anemone clownfish hide",
        "urchin spiny purple rock", "manatee warm spring float", "walrus tusk ice rest",
        "seal pup beach bark", "otter shell rock smash", "beaver dam stream block",
        "platypus bill stream swim", "kangaroo pouch joey", "koala eucalyptus munch",
        "panda bamboo munch sit", "tiger jungle stripe stalk", "lion savanna mane sun",
        "elephant dust bath spray", "giraffe acacia tree reach", "zebra stripe herd dust",
        "rhino mud wallow horn", "hippo river yawn teeth", "gorilla jungle knuckle walk",
        "chimpanzee tool termite", "orangutan vine swing red", "lemur ring tail leap",
        "sloth branch hang slow", "armadillo curl ball roll", "anteater termite mound tongue",
        "bat cave colony hang", "hedge maze green wall lost", "fountain plaza pigeon flock",
        "statue marble weathered crack", "graveyard iron gate fog", "mausoleum stone angel",
        "cemetery candle vigil night", "church bell tower ring", "monastery cliff stone",
        "prayer candle votive row", "incense smoke temple swirl", "monk robe orange walk",
        "nun habit garden quiet", "rabbi torah scroll read", "imam minaret call prayer",
        "shrine torii gate red", "pagoda tier upturn eave", "stupa golden dome prayer flag",
        "mosque dome minaret crescent", "synagogue star stained glass", "cathedral rose window",
        "altar candle gold cross", "pew hymnal open song", "organ loft pipe rank",
        "baptism font water ripple", "confessional booth dark wood", "stained glass saint portrait",
        "fresco ceiling angel host", "mosaic floor geometric pattern", "marble bust roman nose",
        "bronze statue green patina", "iron gate scrollwork vine", "stone gargoyle rain spout",
        "gothic arch flying buttress", "dome oculus light beam", "spiral staircase stone worn",
        "elevator ornate brass gate", "escalator art deco rail", "revolving door chrome hotel",
        "doorman top hat open", "concierge bell desk ring", "hotel key brass tag",
        "suitcase sticker travel worn", "passport stamp border control", "boarding pass barcode scan",
        "window seat wing view cloud", "cockpit instrument glow night", "parachute canopy open blue",
        "life vest orange inflate", "lifeboat orange sea bob", "flare red smoke signal",
        "distress flag wave ship", "message bottle ocean drift", "sos sand beach giant",
        "camp signal mirror flash", "smoke signal hill fire", "bonfire signal beach night",
    ]
