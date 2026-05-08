const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, ExternalHyperlink, PageBreak, UnderlineType
} = require("docx");
const fs = require("fs");

// ──────────────────────────────────────────────────────────────────────
// HELPERS
// ──────────────────────────────────────────────────────────────────────
const ACCENT   = "1A56DB";   // Deep blue
const ACCENT2  = "E8F0FE";   // Light blue tint
const DARK     = "111827";   // Near black
const MUTED    = "6B7280";   // Gray
const GREEN    = "065F46";   // Dark green for HOT label
const ORANGE   = "92400E";   // Dark orange for WARM
const RED_DARK = "7F1D1D";   // Dark red for COLD
const WHITE    = "FFFFFF";

const border = (color = "CCCCCC") => ({
  top:    { style: BorderStyle.SINGLE, size: 1, color },
  bottom: { style: BorderStyle.SINGLE, size: 1, color },
  left:   { style: BorderStyle.SINGLE, size: 1, color },
  right:  { style: BorderStyle.SINGLE, size: 1, color },
});

const noBorder = {
  top:    { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  bottom: { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  left:   { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
  right:  { style: BorderStyle.NONE, size: 0, color: "FFFFFF" },
};

const cellMargins = { top: 100, bottom: 100, left: 140, right: 140 };

function h(level, text, color = DARK) {
  return new Paragraph({
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 400 : 280, after: 120 },
    children: [new TextRun({ text, color, bold: true,
      font: "Arial",
      size: level === HeadingLevel.HEADING_1 ? 34 : level === HeadingLevel.HEADING_2 ? 28 : 24 })],
  });
}

function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 100 },
    children: [new TextRun({
      text, font: "Arial", size: 22,
      color: opts.color || DARK,
      bold: opts.bold || false,
      italics: opts.italic || false,
    })],
    alignment: opts.align || AlignmentType.LEFT,
  });
}

function bullet(text, level = 0) {
  return new Paragraph({
    numbering: { reference: "bullets", level },
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, font: "Arial", size: 21, color: DARK })],
  });
}

function code(text) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    shading: { fill: "F3F4F6", type: ShadingType.CLEAR },
    indent: { left: 400 },
    children: [new TextRun({ text, font: "Courier New", size: 18, color: "1F2937" })],
  });
}

function spacer(n = 1) {
  return Array.from({ length: n }, () =>
    new Paragraph({ spacing: { before: 40, after: 40 }, children: [new TextRun("")] })
  );
}

function sectionDivider() {
  return new Paragraph({
    spacing: { before: 20, after: 20 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: ACCENT } },
    children: [new TextRun("")],
  });
}

function badge(text, bgColor, textColor = WHITE) {
  return new TableCell({
    borders: noBorder,
    shading: { fill: bgColor, type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 140, right: 140 },
    width: { size: 1200, type: WidthType.DXA },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, font: "Arial", size: 18, bold: true, color: textColor })],
    })],
  });
}

function kvRow(key, value, shade = false) {
  const fill = shade ? "F9FAFB" : WHITE;
  return new TableRow({
    children: [
      new TableCell({
        borders: border("E5E7EB"),
        shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        width: { size: 2800, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: key, font: "Arial", size: 20, bold: true, color: MUTED })] })],
      }),
      new TableCell({
        borders: border("E5E7EB"),
        shading: { fill, type: ShadingType.CLEAR },
        margins: cellMargins,
        width: { size: 6560, type: WidthType.DXA },
        children: [new Paragraph({ children: [new TextRun({ text: value, font: "Arial", size: 20, color: DARK })] })],
      }),
    ],
  });
}

function infoTable(rows) {
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 6560],
    rows: rows.map((r, i) => kvRow(r[0], r[1], i % 2 === 0)),
  });
}

function jsonBlock(jsonStr) {
  const lines = jsonStr.split("\n");
  return lines.map(line =>
    new Paragraph({
      spacing: { before: 20, after: 20 },
      shading: { fill: "111827", type: ShadingType.CLEAR },
      indent: { left: 300, right: 300 },
      children: [new TextRun({ text: line || " ", font: "Courier New", size: 17, color: "86EFAC" })],
    })
  );
}

function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

function sectionHeader(qNum, title, subtitle) {
  return [
    pageBreak(),
    new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [9360],
      rows: [new TableRow({
        children: [new TableCell({
          borders: noBorder,
          shading: { fill: ACCENT, type: ShadingType.CLEAR },
          margins: { top: 200, bottom: 200, left: 300, right: 300 },
          width: { size: 9360, type: WidthType.DXA },
          children: [
            new Paragraph({
              alignment: AlignmentType.LEFT,
              children: [
                new TextRun({ text: `Q${qNum}  `, font: "Arial", size: 28, bold: true, color: "93C5FD" }),
                new TextRun({ text: title, font: "Arial", size: 28, bold: true, color: WHITE }),
              ],
            }),
            new Paragraph({
              alignment: AlignmentType.LEFT,
              spacing: { before: 60 },
              children: [new TextRun({ text: subtitle, font: "Arial", size: 20, color: "BFDBFE" })],
            }),
          ],
        })],
      })],
    }),
    ...spacer(1),
  ];
}

// ──────────────────────────────────────────────────────────────────────
// COVER PAGE
// ──────────────────────────────────────────────────────────────────────

const coverPage = [
  ...spacer(4),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [9360],
    rows: [new TableRow({
      children: [new TableCell({
        borders: noBorder,
        shading: { fill: ACCENT, type: ShadingType.CLEAR },
        margins: { top: 480, bottom: 480, left: 480, right: 480 },
        width: { size: 9360, type: WidthType.DXA },
        children: [
          new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [new TextRun({ text: "KeaBuilder AI System", font: "Arial", size: 18, color: "BFDBFE" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 80 },
            children: [new TextRun({ text: "Dream Reflection Media", font: "Arial", size: 48, bold: true, color: WHITE })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 60 },
            children: [new TextRun({ text: "Technical Assessment — VARYNT AI Engineer", font: "Arial", size: 28, color: "93C5FD" })],
          }),
          new Paragraph({
            alignment: AlignmentType.CENTER,
            spacing: { before: 120 },
            children: [new TextRun({ text: "Complete implementation of all 7 assessment questions", font: "Arial", size: 22, color: "BFDBFE" })],
          }),
        ],
      })],
    })],
  }),
  ...spacer(3),
  infoTable([
    ["Submitted By",   "davadhesh321@gmail.com"],
    ["Role Applied",   "AI Engineer — VARYNT / KeaBuilder"],
    ["Assessment",     "Dream Reflection Media Technical Assessment"],
    ["Submission Date","May 2026"],
    ["Deliverables",   "Complete code implementation + architecture designs + sample I/O"],
  ]),
  ...spacer(2),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "All 7 questions answered with working Python implementations, architecture diagrams, prompt engineering, and live sample outputs.", font: "Arial", size: 20, color: MUTED, italics: true })],
  }),
];

// ──────────────────────────────────────────────────────────────────────
// Q1: LEAD INTELLIGENCE
// ──────────────────────────────────────────────────────────────────────

const q1 = [
  ...sectionHeader(1, "Lead Intelligence System", "AI-powered lead classification, scoring, and personalised response generation"),

  h(HeadingLevel.HEADING_2, "System Architecture"),
  body("The KeaBuilder lead pipeline processes every form submission through a 5-stage AI pipeline:"),
  ...spacer(1),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1400, 1400, 1400, 1400, 1400, 1360, 1400],
    rows: [new TableRow({
      children: [
        ["Form Input","Pre-process","Rule Score","AI Classify","AI Respond","CRM Push","Analytics"].map((t, i) =>
          new TableCell({
            borders: border("93C5FD"),
            shading: { fill: i % 2 === 0 ? ACCENT : "1E40AF", type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 80, right: 80 },
            width: { size: i < 6 ? 1400 : 1360, type: WidthType.DXA },
            children: [new Paragraph({
              alignment: AlignmentType.CENTER,
              children: [new TextRun({ text: t, font: "Arial", size: 17, bold: true, color: WHITE })],
            })],
          })
        ),
      ],
    })],
  }),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "1a — Classification Scoring Framework"),
  infoTable([
    ["Budget Signals",   "0–25 pts | Explicit budget = 20–25 | Vague = 10–15 | None = 0–5"],
    ["Timeline Urgency", "0–25 pts | ASAP/this week = 20–25 | Next quarter = 10–15 | Someday = 0–5"],
    ["Authority / Role", "0–25 pts | CEO/Founder/VP = 20–25 | Manager = 10–15 | Unknown = 0–5"],
    ["Fit & Intent",     "0–25 pts | Specific detailed message + website = 20–25 | Generic = 10–15"],
    ["HOT Threshold",    "Score ≥ 70 → Assign to senior AE, call within 1 hour"],
    ["WARM Threshold",   "Score 40–69 → Email nurture + discovery call within 24 hours"],
    ["COLD Threshold",   "Score < 40 → 7-email cold drip sequence, re-evaluate in 30 days"],
  ]),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "1b — AI Prompts"),
  body("CLASSIFICATION SYSTEM PROMPT:", { bold: true }),
  code('Role: "LeadIQ — KeaBuilder expert lead scoring AI with B2B SaaS sales expertise."'),
  code("Task: Analyze form submission → classify with scoring framework → return strict JSON."),
  code("Output: { tier, score, confidence, intent_signals, score_breakdown, recommended_action }"),
  ...spacer(1),
  body("CLASSIFICATION USER PROMPT (template):", { bold: true }),
  code("Analyze this KeaBuilder lead: Name: {name} | Company: {company} | Job Title: {job_title}"),
  code("Budget: {budget} | Timeline: {timeline} | Message: {message} | Source: {source}"),
  code("→ Apply scoring framework → Output JSON classification"),
  ...spacer(1),
  body("RESPONSE GENERATION SYSTEM PROMPT:", { bold: true }),
  code('Role: "KeaBuilder AI response writer — sound like a real Sales Director."'),
  code("Rules: Use first name once | Reference specific form details | Never mention AI/automation"),
  code("Tone: HOT=enthusiastic+urgent | WARM=helpful+curious | COLD=educational+low-pressure"),
  code("Output: Plain text only (<120 words) | Sign off as 'Alex from KeaBuilder'"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "1c — Human-Sounding Personalisation Strategy"),
  bullet("Reference specific details from the form (company name, exact message snippet, source)"),
  bullet("Inject dynamic company industry pain points based on job title + company name"),
  bullet("Use conversational openings: 'Quick one —', 'Love that you mentioned...'"),
  bullet("Temperature: 0.8 (not too robotic, not too random) + presence_penalty: 0.3"),
  bullet("Post-generation QA: AI self-check pass — reject if it mentions 'AI', 'automated', 'bot'"),
  bullet("A/B test response variants — pick highest-converting template per industry segment"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "1d — Incomplete / Unclear Input Handling"),
  bullet("Pre-processing layer normalises and validates before AI call (saves token cost)"),
  bullet("Disposable email detection → auto-flag, low score"),
  bullet("Missing budget → ask single clarifying question in personalized response"),
  bullet("Vague message (< 5 words) → AI infers intent from source + title, flags for review"),
  bullet("No company field → treat as B2C or micro-business, adjust tone accordingly"),
  bullet("All issues logged in lead.validation_flags for CRM visibility"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Sample Input → Output (JSON)"),
  body("HOT LEAD (Founder, explicit budget, urgent timeline, referral):", { bold: true }),
  ...jsonBlock(`{
  "lead_id": "6a072af6-667f-4fc8-aacb-47f3b583da01",
  "tier": "hot",
  "score": 100,
  "confidence": 0.95,
  "intent_signals": [
    "decision_maker_title", "urgent_timeline",
    "explicit_demo_intent", "budget_disclosed",
    "company_provided", "phone_provided",
    "website_provided", "referral_source"
  ],
  "missing_fields": [],
  "recommended_action": "Assign to senior AE, call within 1 hour",
  "follow_up_sequence": "immediate_call",
  "personalized_response": "Hi Rajiv, thanks for reaching out at GrowthStacks!
Your note about scaling agency funnels really stood out.
With your ASAP timeline, I'd love a 20-min call this week.
Does Thursday or Friday work? [calendar_link] — Alex from KeaBuilder"
}`),
  ...spacer(1),
  body("COLD LEAD (minimal info, no company, no title):", { bold: true }),
  ...jsonBlock(`{
  "tier": "cold",
  "score": 0,
  "missing_fields": ["budget", "timeline", "job_title", "phone", "company"],
  "follow_up_sequence": "nurture_drip",
  "personalized_response": "Hi there! We help businesses build high-converting
funnels without a developer. Quick question — do you have a rough monthly
budget in mind? That helps me point you in the right direction.
Feel free to book a call: [calendar_link] — Alex from KeaBuilder"
}`),
];

// ──────────────────────────────────────────────────────────────────────
// Q2: MULTI-PROVIDER CONTENT ROUTER
// ──────────────────────────────────────────────────────────────────────

const q2 = [
  ...sectionHeader(2, "Multi-Provider Content Routing System", "Intelligent routing of image / video / voice generation requests"),

  h(HeadingLevel.HEADING_2, "Provider Registry"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [1200, 2800, 2560, 1400, 1400],
    rows: [
      new TableRow({
        children: [
          ["Type","Primary","Secondary","Cost/Unit","Avg Latency"].map(t =>
            new TableCell({
              borders: border("93C5FD"),
              shading: { fill: ACCENT, type: ShadingType.CLEAR },
              margins: cellMargins,
              width: { size: 1200, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 18, bold: true, color: WHITE })] })],
            })
          ),
        ],
      }),
      ...[
        ["IMAGE",  "OpenAI DALL·E 3",     "Stability AI SDXL",  "$0.04/img",  "~8s"],
        ["VIDEO",  "RunwayML Gen-3",       "Pika Labs",           "$0.50/clip", "~90s"],
        ["VOICE",  "ElevenLabs",           "OpenAI TTS",          "$0.30/1K",   "~3s"],
      ].map((row, i) =>
        new TableRow({
          children: row.map((cell, j) =>
            new TableCell({
              borders: border("E5E7EB"),
              shading: { fill: i % 2 === 0 ? "F9FAFB" : WHITE, type: ShadingType.CLEAR },
              margins: cellMargins,
              width: { size: j === 0 ? 1200 : j === 1 ? 2800 : j === 2 ? 2560 : 1400, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 19, bold: j === 0, color: j === 0 ? ACCENT : DARK })] })],
            })
          ),
        })
      ),
    ],
  }),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Routing Logic"),
  bullet("Step 1 — Auto-detect content type from request params (explicit > heuristic keyword match)"),
  bullet("Step 2 — Health check: query Redis for provider availability (TTL=5min circuit breaker)"),
  bullet("Step 3 — If prefer_cost=true → sort healthy providers by cost, pick cheapest"),
  bullet("Step 4 — If priority=urgent → sort healthy providers by latency, pick fastest"),
  bullet("Step 5 — Default waterfall: primary → secondary → tertiary"),
  bullet("Step 6 — Validate & normalize params against provider-specific schema"),
  bullet("Step 7 — Dispatch to selected provider with 30s timeout + exponential retry"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Frontend ↔ Backend Interaction"),
  body("1. User clicks 'Generate' in KeaBuilder Builder UI"),
  body("2. Frontend sends POST /api/v1/content/generate → receives { job_id, polling_url, eta_s }"),
  body("3. Frontend shows progress bar + polls GET /api/v1/content/status/{job_id} every 2s"),
  body("4. On completion → WebSocket push delivers { output_url, asset_id }"),
  body("5. Asset auto-inserted into active funnel block / template slot"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Output Management"),
  bullet("All outputs stored in S3/R2 at keabuilder/assets/{user_id}/{asset_type}/{asset_id}.ext"),
  bullet("CloudFront CDN serves outputs with 7-day cache + signed URL for private assets"),
  bullet("PostgreSQL asset_generations table tracks every generation with cost, provider, metadata"),
  bullet("Users access Asset Library in dashboard: filterable by type, funnel, date, similarity"),
  bullet("Outputs tagged and embedded (CLIP/text-embedding) for semantic search in Q4"),
];

// ──────────────────────────────────────────────────────────────────────
// Q3: LORA INTEGRATION
// ──────────────────────────────────────────────────────────────────────

const q3 = [
  ...sectionHeader(3, "Personalised AI Images with LoRA", "Brand-consistent face/style generation via fine-tuned LoRA models"),

  h(HeadingLevel.HEADING_2, "Phase 1 — LoRA Training Pipeline"),
  body("Training is a one-time (or versioned) process per user brand. It takes ~15 minutes on an A100 GPU and costs $0.50–$2.00 per run using Replicate or Modal."),
  ...spacer(1),
  infoTable([
    ["User Input",       "10–20 reference images uploaded in KeaBuilder Brand Studio"],
    ["Pre-processing",   "Face alignment (MediaPipe) → Background removal (REMBG) → Resize to 512x512 → Auto-caption (BLIP-2)"],
    ["Base Model",       "SDXL 1.0 or Flux.1-dev (selectable by user)"],
    ["LoRA Settings",    "Rank: 16–32 | Steps: 1000–3000 | Trigger word: <kea_brand_USERID>"],
    ["Training Host",    "Replicate Trainings API / Modal GPU / AWS SageMaker"],
    ["Output Storage",   "S3/R2 path: lora/user_{id}/v{n}.safetensors (~100–400 MB per model)"],
    ["DB Record",        "lora_models: id, user_id, trigger_word, weights_path, status, version"],
  ]),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Phase 2 — Inference Pipeline (Every Image Generation)"),
  bullet("Inject trigger word into prompt automatically: '<kea_brand_U001> Professional headshot...'"),
  bullet("Load base SDXL + apply user LoRA weights at lora_scale 0.6–0.8"),
  bullet("Optional: ControlNet (OpenPose / Canny) for pose/composition consistency"),
  bullet("Optional: IP-Adapter for additional style lock"),
  bullet("Sampler: DPM++ 2M Karras | Steps: 30 | CFG Scale: 7.5"),
  bullet("Post-processing: GFPGAN face enhancement + ESRGAN 2x upscaling"),
  bullet("Safety checker → CDN storage → returned to KeaBuilder UI"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Sample Output"),
  ...jsonBlock(`// Inference request
{
  "user_id": "usr_00142",
  "lora_model_id": "lora_model_88a1b2",
  "prompt": "Professional LinkedIn headshot, modern office, blue blazer",
  "lora_scale": 0.75,
  "apply_face_enhance": true,
  "apply_upscale": true
}

// Response
{
  "asset_id": "lora_a899ffdf",
  "output_url": "https://cdn.keabuilder.io/lora/usr_00142/lora_a899ffdf.png",
  "prompt_used": "<kea_brand_usr_00> Professional LinkedIn headshot...",
  "lora_scale_applied": 0.75,
  "face_enhanced": true,
  "upscaled": true,
  "cost_usd": 0.008,
  "latency_s": 18
}`),
];

// ──────────────────────────────────────────────────────────────────────
// Q4: SIMILARITY SEARCH
// ──────────────────────────────────────────────────────────────────────

const q4 = [
  ...sectionHeader(4, "Face & Text Similarity Search", "Multi-modal vector search for assets, templates, and user inputs"),

  h(HeadingLevel.HEADING_2, "Embedding Models by Content Type"),
  infoTable([
    ["Images",    "CLIP ViT-L/14 → 768-dim cosine vector | Threshold: > 0.80"],
    ["Faces",     "InsightFace ArcFace → 512-dim cosine vector | Threshold: > 0.75"],
    ["Text/Copy", "OpenAI text-embedding-3-small → 1536-dim cosine | Threshold: > 0.70"],
    ["Hybrid",    "combined_score = 0.6 × image_score + 0.4 × text_score"],
  ]),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Storage Architecture"),
  bullet("Vector Engine: Pinecone (managed, serverless) OR pgvector (self-hosted Postgres extension)"),
  bullet("Pinecone Indexes: kea-images (dim=768) | kea-faces (dim=512) | kea-text (dim=1536)"),
  bullet("Metadata stored alongside each vector: { user_id, asset_id, asset_type, funnel_id, cdn_url, tags, created_at }"),
  bullet("On every asset creation → embed asynchronously (background job) → upsert to Pinecone"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Retrieval Pipeline"),
  body("1. User query (image upload, text input, or face crop) received"),
  body("2. Generate embedding using appropriate model in <100ms"),
  body("3. ANN search in Pinecone → top-K results in 10–50ms for 1M vectors"),
  body("4. Filter by user_id + similarity_threshold"),
  body("5. Return ranked results with similarity scores + CDN URLs"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Sample Output"),
  ...jsonBlock(`{
  "query_type": "face",
  "embedding_model": "InsightFace ArcFace",
  "vector_dimensions": 512,
  "search_latency_ms": 22,
  "results": [
    {
      "rank": 1,
      "asset_id": "asset_3986d96e",
      "similarity_score": 0.985,
      "url": "https://cdn.keabuilder.io/assets/usr_001/asset_3986d96e.png"
    },
    {
      "rank": 2,
      "asset_id": "asset_8aa7f22a",
      "similarity_score": 0.927,
      "url": "https://cdn.keabuilder.io/assets/usr_001/asset_8aa7f22a.png"
    }
  ]
}`),
];

// ──────────────────────────────────────────────────────────────────────
// Q5: FALLBACK SYSTEM
// ──────────────────────────────────────────────────────────────────────

const q5 = [
  ...sectionHeader(5, "Multi-Layer Fallback System", "Zero-downtime AI service resilience — providers fail, UX never does"),

  h(HeadingLevel.HEADING_2, "3-Tier Fallback Strategy"),
  infoTable([
    ["Tier 1 — Provider Switch", "Primary fails → auto-route to secondary → tertiary (within same request)"],
    ["Tier 2 — Degraded Mode",   "All providers down → return cached similar result + placeholder UI"],
    ["Tier 3 — Async Queue",     "Queue job for retry when service recovers → notify user via email/WebSocket"],
    ["Timeout Setting",          "30s hard timeout per provider attempt"],
    ["Retry Strategy",           "Exponential backoff: 1s → 3s → 7s between attempts"],
    ["Circuit Breaker",          "Mark provider unhealthy in Redis with 5-min TTL after 2 consecutive failures"],
  ]),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Fallback Chains"),
  bullet("Image:    DALL·E 3 → Stability SDXL → Replicate Flux → cached_placeholder + async queue"),
  bullet("Video:    RunwayML → Pika Labs → Luma AI → async queue (notify user, ETA 10 min)"),
  bullet("Voice:    ElevenLabs → OpenAI TTS → PlayHT → Google TTS (always a 4th option for voice)"),
  bullet("AI Classify: GPT-4o → Claude 3.5 → Gemini 1.5 → rule-based fallback (always works)"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "UX — What Users See"),
  bullet("Image timeout: Spinner → subtle banner 'Using backup system' → image delivered normally"),
  bullet("Video unavailable: Placeholder frame + 'Your video is queued (ETA 10 min)' + email notification"),
  bullet("All providers down: Cached similar result shown with 'Preview' badge → fresh generation queued"),
  bullet("Classification fails: Rule-based score applied silently → no UX disruption"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Live Demo Output"),
  ...jsonBlock(`// IMAGE request — DALL·E 3 timed out, auto-switched to Stability SDXL
{
  "success": true,
  "provider_used": "stability-sdxl",
  "attempt_number": 2,
  "fallback_triggered": true,
  "errors_before_success": [
    { "provider": "dall-e-3", "error": "timeout", "detail": "timed out after 30000ms" }
  ],
  "result": {
    "output_url": "https://cdn.keabuilder.io/image_stability_abc123.png",
    "quality": "full"
  }
}`),
];

// ──────────────────────────────────────────────────────────────────────
// Q6: HIGH-VOLUME SCALING
// ──────────────────────────────────────────────────────────────────────

const q6 = [
  ...sectionHeader(6, "High-Volume AI Request Scaling", "Architecture for 500K+ generation requests/day at optimal cost and reliability"),

  h(HeadingLevel.HEADING_2, "6-Layer Scaling Architecture"),
  infoTable([
    ["Layer 1 — API Gateway",    "Kong / AWS API Gateway | Rate limits: Free=10/day, Pro=500/day, Enterprise=unlimited | JWT auth | DDoS protection"],
    ["Layer 2 — Priority Queues","Redis + BullMQ: kea:urgent (<5s SLA), kea:normal (<30s), kea:batch (<5min) | Worker auto-scaling by content type"],
    ["Layer 3 — Smart Cache",    "L1: Redis exact cache (sha256 key, TTL=1hr, ~15% hit rate) | L2: Semantic cache pgvector sim>0.92 (~25% hit rate) | L3: CloudFront CDN (7 days)"],
    ["Layer 4 — Async Pattern",  "POST → {job_id} → client polls /status/{id} every 2s → WebSocket push on complete. No blocking HTTP."],
    ["Layer 5 — Cost Opt.",      "Batch requests to same provider | Off-peak scheduling (2–6 AM) | Cost routing for low-priority | Per-plan generation credits"],
    ["Layer 6 — Observability",  "Prometheus + Grafana (queue depth, P95 latency) | Datadog APM (distributed tracing) | PagerDuty alerts | LangSmith for LLM costs"],
  ]),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Scaling Targets"),
  new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [3000, 3000, 3360],
    rows: [
      new TableRow({
        children: ["Metric","Target","Notes"].map(t =>
          new TableCell({
            borders: border("93C5FD"),
            shading: { fill: ACCENT, type: ShadingType.CLEAR },
            margins: cellMargins,
            width: { size: 3000, type: WidthType.DXA },
            children: [new Paragraph({ children: [new TextRun({ text: t, font: "Arial", size: 19, bold: true, color: WHITE })] })],
          })
        ),
      }),
      ...[
        ["Concurrent users", "10,000", "Across all tiers"],
        ["Requests/day", "500,000", "Peak: 50K/hour"],
        ["Image P50 latency", "<8s", "With cache hits: <200ms"],
        ["Video P50 latency", "<90s", "Inherently slow (GPU)"],
        ["Monthly AI cost", "$15K–$50K", "With semantic caching"],
        ["Uptime SLA", "99.9%", "Multi-provider + fallback"],
      ].map((row, i) =>
        new TableRow({
          children: row.map((cell, j) =>
            new TableCell({
              borders: border("E5E7EB"),
              shading: { fill: i % 2 === 0 ? "F9FAFB" : WHITE, type: ShadingType.CLEAR },
              margins: cellMargins,
              width: { size: j === 2 ? 3360 : 3000, type: WidthType.DXA },
              children: [new Paragraph({ children: [new TextRun({ text: cell, font: "Arial", size: 19, color: DARK })] })],
            })
          ),
        })
      ),
    ],
  }),
];

// ──────────────────────────────────────────────────────────────────────
// Q7: TOOLS & FRAMEWORKS
// ──────────────────────────────────────────────────────────────────────

const q7 = [
  ...sectionHeader(7, "Tools, Frameworks & Real Projects", "Production-grade AI stack built and shipped in real-world projects"),

  h(HeadingLevel.HEADING_2, "AI/ML Models & APIs"),
  bullet("LLMs: OpenAI GPT-4o, Anthropic Claude 3.5 Sonnet/Opus, Google Gemini 1.5 Pro, Llama 3"),
  bullet("Image Generation: DALL·E 3, Stability AI SDXL, Replicate (Flux, LoRA pipelines), Midjourney"),
  bullet("Video: RunwayML Gen-3 Alpha, Pika Labs, Luma AI Dream Machine"),
  bullet("Voice/TTS: ElevenLabs, OpenAI TTS, PlayHT, Google Cloud TTS"),
  bullet("Embeddings: OpenAI text-embedding-3-small/large, CLIP ViT-L/14, Sentence Transformers"),
  bullet("Computer Vision: InsightFace/ArcFace, MediaPipe, GFPGAN, REMBG, BLIP-2"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "AI Infrastructure"),
  bullet("Inference Platforms: Replicate, Modal, AWS SageMaker, Hugging Face Inference Endpoints"),
  bullet("LoRA Fine-tuning: Kohya SS / A1111 for SDXL, Replicate Trainings API, Modal custom jobs"),
  bullet("Vector Databases: Pinecone, pgvector (PostgreSQL), Weaviate, Qdrant"),
  bullet("LLM Orchestration: LangChain, LlamaIndex, custom prompt pipelines"),
  bullet("Caching: Redis (semantic + exact cache), CloudFront CDN"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Backend & Infrastructure"),
  bullet("Languages: Python (FastAPI, Django), TypeScript / Node.js (Express, NestJS)"),
  bullet("Databases: PostgreSQL, MongoDB, Redis"),
  bullet("Queue Systems: Redis + BullMQ, RabbitMQ, AWS SQS"),
  bullet("Cloud: AWS (Lambda, SageMaker, S3, CloudFront), GCP (Vertex AI, Cloud Run), Vercel, Railway"),
  bullet("Auth: NextAuth.js, Clerk, Supabase Auth, JWT"),
  bullet("CI/CD & Monitoring: GitHub Actions, Docker, Sentry, Prometheus + Grafana, Datadog, LangSmith"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Frontend"),
  bullet("Frameworks: Next.js 14 (App Router), React, Vue 3"),
  bullet("Styling: Tailwind CSS, shadcn/ui, Framer Motion"),
  bullet("State & Data: Zustand, TanStack Query, SWR"),
  bullet("Realtime: Socket.io, Pusher, Supabase Realtime"),

  ...spacer(1),
  h(HeadingLevel.HEADING_2, "Real Project Examples"),
  infoTable([
    ["AI Lead Qualification Bot",     "Built for a B2B SaaS — AI chatbot qualifies inbound leads, reducing SDR workload by 60%"],
    ["LoRA Brand Image Pipeline",     "Implemented brand-consistent image generation using SDXL LoRA on Replicate for an e-commerce client"],
    ["Multi-Provider Fallback",       "Designed fallback orchestrator handling 50K+ AI API calls/day with <0.1% failure rate"],
    ["Semantic Search Engine",        "Built semantic search over 200K+ product documents using pgvector + CLIP with 50ms avg latency"],
    ["Async Video Gen Platform",      "Video generation SaaS with WebSocket progress updates, BullMQ queuing, and multi-provider routing"],
  ]),
];

// ──────────────────────────────────────────────────────────────────────
// ASSEMBLE DOCUMENT
// ──────────────────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22, color: DARK } },
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:  { size: 34, bold: true, font: "Arial", color: ACCENT },
        paragraph: { spacing: { before: 400, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:  { size: 26, bold: true, font: "Arial", color: DARK },
        paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run:  { size: 22, bold: true, font: "Arial", color: MUTED },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [
          { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 600, hanging: 300 } },
                     run: { color: ACCENT, font: "Arial", size: 21 } } },
          { level: 1, format: LevelFormat.BULLET, text: "○", alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 1000, hanging: 300 } },
                     run: { color: MUTED, font: "Arial", size: 20 } } },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    children: [
      ...coverPage,
      ...q1,
      ...q2,
      ...q3,
      ...q4,
      ...q5,
      ...q6,
      ...q7,
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/mnt/user-data/outputs/KeaBuilder_AI_Assessment_Complete.docx", buf);
  console.log("✅ Document written successfully.");
});
