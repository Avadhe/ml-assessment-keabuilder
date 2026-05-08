"""
KeaBuilder Multi-Provider Content Routing System
Q2: Intelligent routing of image/video/voice generation requests

Architecture:
  KeaBuilder UI → Content Router → Provider Registry → Output Manager → Asset Storage
"""

import json
import time
import random
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import datetime


# ─────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────

class ContentType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    VOICE = "voice"

class OutputStatus(str, Enum):
    PENDING   = "pending"
    QUEUED    = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED    = "failed"

# Provider registry — configure via environment/database in production
PROVIDER_CONFIG = {
    ContentType.IMAGE: {
        "primary":   {"name": "OpenAI DALL·E 3",     "endpoint": "https://api.openai.com/v1/images/generations",   "cost_per_unit": 0.04,  "avg_latency_s": 8},
        "secondary": {"name": "Stability AI SDXL",   "endpoint": "https://api.stability.ai/v2beta/stable-image",   "cost_per_unit": 0.003, "avg_latency_s": 5},
        "tertiary":  {"name": "Replicate Flux",       "endpoint": "https://api.replicate.com/v1/predictions",        "cost_per_unit": 0.005, "avg_latency_s": 12},
        "supported_params": ["prompt", "negative_prompt", "width", "height", "style", "quality", "n"],
        "max_resolution": "1792x1024",
    },
    ContentType.VIDEO: {
        "primary":   {"name": "RunwayML Gen-3 Alpha", "endpoint": "https://api.runwayml.com/v1/image_to_video",     "cost_per_unit": 0.50,  "avg_latency_s": 90},
        "secondary": {"name": "Pika Labs",             "endpoint": "https://api.pika.art/v1/generate",               "cost_per_unit": 0.30,  "avg_latency_s": 120},
        "tertiary":  {"name": "Luma AI Dream Machine", "endpoint": "https://api.lumalabs.ai/dream-machine/v1",       "cost_per_unit": 0.40,  "avg_latency_s": 60},
        "supported_params": ["prompt", "duration_seconds", "aspect_ratio", "fps", "image_url"],
        "max_duration_s": 10,
    },
    ContentType.VOICE: {
        "primary":   {"name": "ElevenLabs",           "endpoint": "https://api.elevenlabs.io/v1/text-to-speech",    "cost_per_unit": 0.30,  "avg_latency_s": 3},
        "secondary": {"name": "OpenAI TTS",           "endpoint": "https://api.openai.com/v1/audio/speech",          "cost_per_unit": 0.015, "avg_latency_s": 2},
        "tertiary":  {"name": "PlayHT",               "endpoint": "https://api.play.ht/api/v2/tts",                  "cost_per_unit": 0.05,  "avg_latency_s": 4},
        "supported_params": ["text", "voice_id", "stability", "similarity_boost", "style", "output_format"],
        "max_chars": 5000,
    },
}


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class ContentRequest:
    request_id: str
    user_id: str
    funnel_id: str
    content_type: ContentType
    prompt: str
    params: dict = field(default_factory=dict)
    priority: str = "normal"        # low | normal | high | urgent
    prefer_cost: bool = False       # if True, pick cheapest provider
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

@dataclass
class RoutingDecision:
    request_id: str
    content_type: ContentType
    selected_provider: str
    provider_tier: str              # primary | secondary | tertiary
    reason: str
    estimated_cost_usd: float
    estimated_latency_s: int
    queue_position: Optional[int]
    normalized_params: dict

@dataclass
class ContentOutput:
    request_id: str
    user_id: str
    content_type: ContentType
    status: OutputStatus
    provider_used: str
    output_url: Optional[str]
    output_metadata: dict
    cost_usd: float
    processing_time_s: float
    asset_id: Optional[str]
    storage_path: Optional[str]
    error: Optional[str]
    created_at: str
    completed_at: Optional[str]


# ─────────────────────────────────────────────
# ROUTING LOGIC
# ─────────────────────────────────────────────

class ContentRouter:
    """
    Central routing engine that:
    1. Detects content type
    2. Validates & normalizes parameters
    3. Selects optimal provider
    4. Manages output lifecycle
    """

    def __init__(self):
        self.provider_health = {
            ContentType.IMAGE: {"primary": True, "secondary": True, "tertiary": True},
            ContentType.VIDEO: {"primary": True, "secondary": True, "tertiary": True},
            ContentType.VOICE: {"primary": True, "secondary": False, "tertiary": True},  # Simulate one down
        }
        self._queue = []

    def detect_content_type(self, request_data: dict) -> ContentType:
        """Auto-detect content type from request metadata if not explicitly set."""
        explicit = request_data.get("content_type", "").lower()
        if explicit in [ct.value for ct in ContentType]:
            return ContentType(explicit)

        # Heuristic detection from prompt / params
        prompt = request_data.get("prompt", "").lower()
        if any(kw in prompt for kw in ["video", "animate", "motion", "seconds long", "footage"]):
            return ContentType.VIDEO
        if any(kw in prompt for kw in ["speak", "voice", "say this", "narrate", "audio", "tts"]):
            return ContentType.VOICE
        return ContentType.IMAGE  # Default

    def validate_and_normalize(self, content_type: ContentType, params: dict) -> tuple[dict, list[str]]:
        """Validate params against provider requirements and normalize to common schema."""
        config = PROVIDER_CONFIG[content_type]
        supported = config["supported_params"]
        warnings = []

        normalized = {k: v for k, v in params.items() if k in supported}

        # Apply content-type specific defaults and limits
        if content_type == ContentType.IMAGE:
            normalized.setdefault("width", 1024)
            normalized.setdefault("height", 1024)
            normalized.setdefault("quality", "standard")
            normalized.setdefault("n", 1)
            if normalized.get("n", 1) > 4:
                normalized["n"] = 4
                warnings.append("n capped at 4 per request")

        elif content_type == ContentType.VIDEO:
            normalized.setdefault("duration_seconds", 5)
            normalized.setdefault("aspect_ratio", "16:9")
            normalized.setdefault("fps", 24)
            if normalized.get("duration_seconds", 5) > config["max_duration_s"]:
                normalized["duration_seconds"] = config["max_duration_s"]
                warnings.append(f"Duration capped at {config['max_duration_s']}s")

        elif content_type == ContentType.VOICE:
            normalized.setdefault("output_format", "mp3")
            normalized.setdefault("stability", 0.5)
            normalized.setdefault("similarity_boost", 0.75)
            text = params.get("text", "")
            if len(text) > config["max_chars"]:
                warnings.append(f"Text truncated to {config['max_chars']} chars")
                normalized["text"] = text[:config["max_chars"]]

        return normalized, warnings

    def select_provider(self, content_type: ContentType, prefer_cost: bool, priority: str) -> tuple[str, str]:
        """
        Provider selection strategy:
        1. Check health of primary → fallback to secondary → tertiary
        2. If prefer_cost=True → sort by cost, pick cheapest healthy provider
        3. If priority=urgent → pick fastest healthy provider
        """
        config = PROVIDER_CONFIG[content_type]
        health = self.provider_health[content_type]

        tiers = ["primary", "secondary", "tertiary"]

        if prefer_cost:
            # Sort healthy providers by cost
            healthy_tiers = [(t, config[t]["cost_per_unit"]) for t in tiers if health.get(t, False)]
            if healthy_tiers:
                healthy_tiers.sort(key=lambda x: x[1])
                tier = healthy_tiers[0][0]
                return config[tier]["name"], tier

        if priority == "urgent":
            # Sort healthy providers by latency
            healthy_tiers = [(t, config[t]["avg_latency_s"]) for t in tiers if health.get(t, False)]
            if healthy_tiers:
                healthy_tiers.sort(key=lambda x: x[1])
                tier = healthy_tiers[0][0]
                return config[tier]["name"], tier

        # Default: primary-first waterfall
        for tier in tiers:
            if health.get(tier, False):
                return config[tier]["name"], tier

        raise RuntimeError(f"No healthy providers available for {content_type}")

    def route(self, request: ContentRequest) -> RoutingDecision:
        """Main routing entry point — returns a full routing decision."""
        normalized_params, warnings = self.validate_and_normalize(
            request.content_type, request.params
        )

        provider_name, provider_tier = self.select_provider(
            request.content_type, request.prefer_cost, request.priority
        )

        config = PROVIDER_CONFIG[request.content_type][provider_tier]
        queue_pos = len(self._queue) + 1 if request.priority == "low" else None

        return RoutingDecision(
            request_id=request.request_id,
            content_type=request.content_type,
            selected_provider=provider_name,
            provider_tier=provider_tier,
            reason=self._explain_routing(request, provider_tier, warnings),
            estimated_cost_usd=config["cost_per_unit"],
            estimated_latency_s=config["avg_latency_s"],
            queue_position=queue_pos,
            normalized_params=normalized_params,
        )

    def _explain_routing(self, request: ContentRequest, tier: str, warnings: list) -> str:
        reason = f"Selected {tier} provider"
        if tier != "primary":
            reason += " (primary unhealthy/unavailable)"
        if request.prefer_cost:
            reason += "; cost-optimized mode"
        if request.priority == "urgent":
            reason += "; urgent priority — fastest provider chosen"
        if warnings:
            reason += f"; warnings: {', '.join(warnings)}"
        return reason

    def simulate_execution(self, decision: RoutingDecision) -> ContentOutput:
        """Simulate provider execution and return output object."""
        import uuid
        start = time.time()

        # Simulate variable latency
        simulated_time = decision.estimated_latency_s * random.uniform(0.7, 1.3)
        # time.sleep(simulated_time)  # Skip actual sleep in demo

        asset_id = f"asset_{uuid.uuid4().hex[:8]}"
        content_type = decision.content_type

        ext_map = {ContentType.IMAGE: "png", ContentType.VIDEO: "mp4", ContentType.VOICE: "mp3"}
        ext = ext_map[content_type]
        storage_path = f"keabuilder/assets/{content_type.value}/{asset_id}.{ext}"
        cdn_url = f"https://cdn.keabuilder.io/{storage_path}"

        return ContentOutput(
            request_id=decision.request_id,
            user_id="usr_demo",
            content_type=content_type,
            status=OutputStatus.COMPLETED,
            provider_used=decision.selected_provider,
            output_url=cdn_url,
            output_metadata={
                "asset_id": asset_id,
                "provider": decision.selected_provider,
                "provider_tier": decision.provider_tier,
                "params_used": decision.normalized_params,
                "simulated_latency_s": round(simulated_time, 2),
            },
            cost_usd=decision.estimated_cost_usd,
            processing_time_s=round(time.time() - start + simulated_time, 2),
            asset_id=asset_id,
            storage_path=storage_path,
            error=None,
            created_at=datetime.utcnow().isoformat() + "Z",
            completed_at=datetime.utcnow().isoformat() + "Z",
        )


# ─────────────────────────────────────────────
# FRONTEND ↔ BACKEND CONTRACT (API Schema)
# ─────────────────────────────────────────────

FRONTEND_API_SCHEMA = {
    "POST /api/v1/content/generate": {
        "description": "KeaBuilder Builder UI calls this endpoint when user clicks Generate",
        "request_body": {
            "user_id": "string (required)",
            "funnel_id": "string (required)",
            "content_type": "image | video | voice",
            "prompt": "string (required)",
            "params": {
                "__image__": {"width": "int", "height": "int", "style": "string", "quality": "standard|hd", "n": "int 1-4"},
                "__video__": {"duration_seconds": "int 1-10", "aspect_ratio": "16:9|9:16|1:1", "fps": "24|30"},
                "__voice__": {"text": "string", "voice_id": "string", "output_format": "mp3|wav|ogg"},
            },
            "priority": "low | normal | high | urgent",
            "prefer_cost": "boolean",
        },
        "response_body": {
            "request_id": "string",
            "status": "queued | processing",
            "estimated_latency_s": "int",
            "estimated_cost_usd": "float",
            "polling_url": "/api/v1/content/status/{request_id}",
        }
    },
    "GET /api/v1/content/status/{request_id}": {
        "description": "Frontend polls this until status = completed | failed",
        "response_body": {
            "request_id": "string",
            "status": "pending | queued | processing | completed | failed",
            "output_url": "string (CDN URL, available when completed)",
            "asset_id": "string",
            "cost_usd": "float",
            "error": "string | null",
        }
    },
    "GET /api/v1/content/assets": {
        "description": "Fetch all generated assets for a user's funnel",
        "query_params": {"user_id": "string", "funnel_id": "string", "content_type": "image|video|voice"},
    }
}


# ─────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uuid, json
    from dataclasses import asdict

    router = ContentRouter()

    test_requests = [
        ContentRequest(
            request_id=str(uuid.uuid4()),
            user_id="usr_001",
            funnel_id="fkb_004",
            content_type=ContentType.IMAGE,
            prompt="A sleek SaaS dashboard UI on a MacBook Pro, modern minimalist, dark theme",
            params={"width": 1792, "height": 1024, "quality": "hd", "style": "vivid"},
            priority="normal",
        ),
        ContentRequest(
            request_id=str(uuid.uuid4()),
            user_id="usr_002",
            funnel_id="fkb_001",
            content_type=ContentType.VIDEO,
            prompt="Cinematic product reveal animation for a SaaS platform, 5 seconds",
            params={"duration_seconds": 5, "aspect_ratio": "16:9", "fps": 24},
            priority="urgent",
        ),
        ContentRequest(
            request_id=str(uuid.uuid4()),
            user_id="usr_003",
            funnel_id="fkb_007",
            content_type=ContentType.VOICE,
            prompt="Professional voiceover",
            params={
                "text": "Welcome to KeaBuilder — the smartest way to build marketing funnels that convert.",
                "voice_id": "rachel_professional",
                "output_format": "mp3",
            },
            priority="normal",
            prefer_cost=True,  # Voice primary is ElevenLabs (expensive), pick cheapest
        ),
    ]

    print("=" * 70)
    print("KeaBuilder Content Router — Demo Run")
    print("=" * 70)

    for req in test_requests:
        print(f"\n{'─'*70}")
        print(f"🎨 REQUEST: {req.content_type.value.upper()} | {req.priority.upper()} priority")
        print(f"   Prompt: {req.prompt[:60]}...")

        decision = router.route(req)
        output = router.simulate_execution(decision)

        result = {
            "routing_decision": asdict(decision),
            "output": asdict(output),
        }
        print(json.dumps(result, indent=2, default=str))

    print("\n" + "=" * 70)
    print("API Schema (Frontend ↔ Backend contract):")
    print(json.dumps(FRONTEND_API_SCHEMA, indent=2))
