# SpatialFlux

An enterprise-grade, high-throughput spatial video ingestion plane engineered to bridge physical edge devices (robotic arms, autonomous rovers, facility cameras) with centralized Vision-Language-Action (VLA) foundation models.

`SpatialFlux` minimizes network transit footprint and cloud inference overhead by introducing an intelligent frame-routing control plane that shifts localized perception to the edge, running zero-copy frame calculations to compute structural drift before triggering upstream multimodal evaluations.

---

## 🏗️ Data Plane Pipeline
[Robotic Sensors / RTSP Stream]
│
▼ (Raw Video Frame Buffers)
┌────────────────────────────────────────────────────────┐
│                   SPATIALFLUX EDGE CORE                │
│                                                        │
│   ┌────────────────────────┐    ┌────────────────────┐ │
│   │ GStreamer Shared Mem   │ ──►│ Local SVM Layer    │ │
│   │ (Zero-Copy Intercept)  │    │ (YOLOv11 / RT-DETR)│ │
│   └────────────────────────┘    └─────────┬──────────┘ │
│                                           │            │
│                                           ▼            │
│                                 ┌────────────────────┐ │
│                                 │ Temporal Scene Graph│ │
│                                 └─────────┬──────────┘ │
└───────────────────────────────────────────┼────────────┘
│ (Structural Drift Detected)
▼
┌──────────────────────────────────┐
│    Cloud VLA Orchestrator Mesh    │
│ (Selective Region of Interest)   │
└──────────────────────────────────┘


---

## ⚖️ Edge Perception vs. Cloud Action

`SpatialFlux` dynamically balances localized computing parameters against multi-modal processing boundaries to ensure predictable network topologies:

| Engineering Dimension | Edge Perception Layer | Cloud VLA Reasoning Plane |
| :--- | :--- | :--- |
| **Target Engine** | Local Small Vision Models (SVM) | Multi-modal Vision Reasoners |
| **Latency Targets** | Sub-5 milliseconds | 150 to 450 milliseconds |
| **Compute Footprint** | Discrete NVIDIA Jetson / Mobile NVDEC | Distributed Cloud H100/MI300X Clusters |
| **Data Flow Role** | Frame decoding, optical flow tracking, IoU overlap analysis. | High-level goal planning, step-by-step mechanical instruction generation. |
| **Trigger Policy** | Runs continuously at 30-60 frames per second. | Invoked only when local spatial state arrays mismatch baseline trajectories. |

---

## 🚀 Key Architectural Modules

*   **Zero-Copy Frame Interception:** Leverages shared memory bindings (`shm`) via GStreamer pipelines, passing raw hardware decoder pointers straight into inference engines without wasting memory cycles duplication.
*   **Dynamic Region-of-Interest (RoI) Cropping:** Rather than transmitting complete 4K frames over fragile cellular or satellite uplinks, the engine dynamically crops, packs, and serializes localized bounding coordinate arrays.
*   **Temporal Drift Trackers:** Maintains a rolling matrix calculation of physical target positions. If an object moves outside of expected bounds, a network-level event is instantly generated to update the upstream agent message fabric.



---

## 📍 Region-Level Motion Localization

The local drift evaluator is not limited to a single global drift scalar. On every frame after the
first, `LocalSpatialEvaluator` also computes a set of **drift regions**: bounding boxes over the
areas of the frame where motion actually occurred, in addition to the existing frame-wide
`drift_score`.

This works by taking the same grayscale absolute-difference mask used for the scalar score,
dilating it slightly (`cv2.dilate`) to merge nearby scattered motion pixels into coherent blobs,
then running `cv2.findContours` over the result to recover bounding boxes. Boxes below a
configurable minimum area ratio are dropped as noise, and the surviving boxes are returned sorted
largest-first, capped at a configurable maximum count.

`POST /v1/spatial/ingest` includes these as a `drift_regions` list, each entry shaped as
`{"x", "y", "width", "height", "area_ratio"}` (an empty list on the first frame, where there is no
baseline to diff against).

**Scope, honestly stated:** this is classical, deterministic computer vision — a frame-difference
mask plus contour analysis — running fully offline with no model weights. It tells you *where*
something changed in the frame. It has no idea *what* changed: it cannot tell a person from a box
from a shadow, and it does not classify, recognize, or track objects across frames. True
object detection/classification (e.g. a YOLO-class model) remains a roadmap item, not something
this repository currently does or claims to do.

---

For a look at the technical shifts shaping these roles, see the Computer Vision Trends 2026: The Age of Agentic Intelligence breakdown. This video outlines how industry architecture has completely moved away from static classification maps to Vision-Language-Action models requiring real-time high-throughput streaming pipelines.
