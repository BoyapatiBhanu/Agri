"""
main.py -- Agricultural Multimodal RAG Pipeline (Interactive CLI)

8-Layer Pipeline:
  1. Input        : image + text + document
  2. Embedding    : CLIP + MiniLM -> 512-d vectors
  3. Fusion       : weighted combination
  4. Cross-Modal  : transformer attention
  5. Multi-Vector : k=8 sub-vectors
  6. Retrieval    : FAISS top-k search
  7. RAG          : evidence-conditioned generation
  8. Output       : diagnosis + explanation + treatment + confidence

Usage:
    python main.py                  # interactive mode (user types queries)
    python main.py --demo           # run preset demo queries
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_query(pipeline, text="", image_path=None, document=""):
    """Run a single query through the pipeline and print results."""
    from PIL import Image

    image = None
    if image_path and image_path.strip():
        path = image_path.strip().strip('"').strip("'").strip("<").strip(">").strip()
        if not os.path.exists(path):
            print(f"  [ERROR] Path not found: {path}")
            return None
        # If it's a directory, pick the first image file inside
        if os.path.isdir(path):
            IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")
            files = sorted([f for f in os.listdir(path) if f.lower().endswith(IMG_EXTS)])
            if not files:
                print(f"  [ERROR] No image files found in: {path}")
                return None
            path = os.path.join(path, files[0])
            print(f"  Picked image from folder: {os.path.basename(path)} ({len(files)} total images)")
        try:
            image = Image.open(path).convert("RGB")
            print(f"  Image loaded: {path} ({image.size[0]}x{image.size[1]})")
        except Exception as e:
            print(f"  [ERROR] Cannot open image: {e}")
            return None

    print()
    start = time.time()
    result = pipeline.predict(image=image, text=text, document=document)
    elapsed = time.time() - start

    print("=" * 70)
    print(f"  DIAGNOSIS RESULT  (inference: {elapsed:.2f}s)")
    print("=" * 70)

    conf_pct = f"{result.confidence * 100:.1f}%"
    severity = result.severity.upper()
    print(f"  Confidence : {conf_pct}")
    print(f"  Severity   : {severity}")
    print("-" * 70)

    print("\n[DIAGNOSIS]")
    print(result.diagnosis)

    print("\n[EXPLANATION]")
    print(result.explanation)

    print("\n[RECOMMENDATION]")
    print(result.recommendation)

    if result.evidence_sources:
        print("\n[EVIDENCE SOURCES]")
        print(f"  {'Disease':<35} {'Crop':<15} {'Score'}")
        print(f"  {'-'*35} {'-'*15} {'-'*8}")
        for src in result.evidence_sources:
            score = f"{src['score']:.4f}"
            print(f"  {src['disease']:<35} {src['crop']:<15} {score}")

    # -- Evaluation Metrics --
    m = result.metrics
    if m:
        print("\n" + "=" * 70)
        print("  EVALUATION METRICS")
        print("=" * 70)

        print(f"\n  Modalities Used   : {', '.join(m.get('modalities_used', []))}")
        print(f"  Top-K Retrieved   : {m.get('top_k', 0)}")

        print(f"\n  --- Retrieval Scores ---")
        print(f"  Top-1 Score       : {m.get('top1_score', 0):.4f}")
        print(f"  Mean Score        : {m.get('mean_score', 0):.4f}")
        print(f"  Min Score         : {m.get('min_score', 0):.4f}")
        print(f"  Max Score         : {m.get('max_score', 0):.4f}")
        print(f"  Score Spread      : {m.get('score_spread', 0):.4f}")
        print(f"  Score Margin(1-2) : {m.get('score_margin', 0):.4f}")

        print(f"\n  --- Retrieval Quality ---")
        print(f"  Precision@K       : {m.get('precision_at_k', 0):.4f}  ({m.get('precision_at_k', 0)*100:.1f}%)")
        print(f"  Recall@K          : {m.get('recall_at_k', 0):.4f}  ({m.get('recall_at_k', 0)*100:.1f}%)")
        print(f"  F1@K              : {m.get('f1_at_k', 0):.4f}  ({m.get('f1_at_k', 0)*100:.1f}%)")
        print(f"  MRR               : {m.get('mrr', 0):.4f}")
        print(f"  NDCG@K            : {m.get('ndcg_at_k', 0):.4f}")
        print(f"  Fusion Score      : {m.get('fusion_score', 0):.4f}")

        timings = m.get("timings", {})
        if timings:
            print(f"\n  --- Latency Breakdown ---")
            for layer in ["embedding_text", "embedding_image", "embedding_doc",
                          "fusion", "cross_modal", "multi_vector", "retrieval", "rag_generation"]:
                if layer in timings:
                    label = layer.replace("_", " ").title()
                    print(f"  {label:<20}: {timings[layer]*1000:>8.2f} ms")
            print(f"  {'Total':<20}: {timings.get('total', 0)*1000:>8.2f} ms")

    print("=" * 70)
    print()
    return result


def interactive_mode(pipeline):
    """Interactive loop: user types symptoms, gets diagnosis."""
    print("\n" + "=" * 70)
    print("  AGRICULTURAL DISEASE DIAGNOSIS - INTERACTIVE MODE")
    print("=" * 70)
    print()
    print("  Commands:")
    print("    - Type your symptoms to get a diagnosis")
    print("    - Type 'image:<path>' to include an image")
    print("    - Type 'doc:<text>' to include document context")
    print("    - Type 'quit' or 'exit' to stop")
    print("    - Type 'help' for examples")
    print()

    while True:
        print("-" * 70)
        try:
            user_input = input("  Describe symptoms: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            print("  Please enter a symptom description.")
            continue

        lower = user_input.lower()

        if lower in ("quit", "exit", "q"):
            print("  Goodbye!")
            break

        if lower == "help":
            print_help()
            continue

        # Parse input for image/doc prefixes
        text = ""
        image_path = None
        document = ""

        parts = user_input.split("|")
        for part in parts:
            part = part.strip()
            if part.lower().startswith("image:"):
                image_path = part[6:].strip()
            elif part.lower().startswith("doc:"):
                document = part[4:].strip()
            else:
                text = part

        if not text and not image_path and not document:
            print("  Please enter a valid query.")
            continue

        print()
        run_query(pipeline, text=text, image_path=image_path, document=document)
        print()


def print_help():
    """Print help with example queries."""
    print()
    print("  EXAMPLE QUERIES:")
    print("  " + "-" * 50)
    print("  My rice leaves have brown spots with gray centers")
    print("  Tomato leaves show water-soaked dark spots")
    print("  Wheat has orange-brown pustules on leaves")
    print("  Lower leaves turning yellow, stunted growth")
    print("  Banana plant wilting with brown stem inside")
    print("  Coffee leaves have orange powder underneath")
    print()
    print("  WITH IMAGE:")
    print("  leaf spots on my crop | image:C:\\photos\\leaf.jpg")
    print()
    print("  WITH DOCUMENT:")
    print("  plant is wilting | doc:lab report shows fungal infection")
    print()
    print("  MULTIMODAL:")
    print("  what disease? | image:leaf.jpg | doc:soil pH is 7.8")
    print()


def demo_queries(pipeline):
    """Run preset demo queries."""
    queries = [
        "My rice leaves have diamond-shaped gray lesions with brown borders",
        "Tomato plant leaves show large dark water-soaked spots with white fuzz underneath",
        "My wheat crop has orange-brown pustules scattered on leaf surfaces",
        "Lower leaves of my plants are turning yellow. Stunted growth and thin stems",
        "Young leaves show interveinal chlorosis with veins remaining green. Soil is alkaline",
        "Banana plant has yellowing lower leaves and brown vascular discoloration in pseudostem",
        "Coffee plant leaves show yellow-orange powdery spots on the underside",
        "Corn leaves show large cigar-shaped gray-green lesions progressing from lower leaves",
    ]

    print("\n" + "=" * 70)
    print("  RUNNING DEMO QUERIES")
    print("=" * 70)

    for i, q in enumerate(queries, 1):
        print(f"\n{'#' * 70}")
        print(f"  Query {i}/{len(queries)}")
        print(f"  Text: {q}")
        print(f"{'#' * 70}\n")
        run_query(pipeline, text=q)

    print("\n  Demo complete.")


def main():
    # Check if FAISS index exists
    from config import FAISS_INDEX_PATH
    meta_path = os.path.join(FAISS_INDEX_PATH, "metadata.pkl")
    if not os.path.exists(meta_path):
        print("[ERROR] FAISS index not found. Run 'python build_index.py' first.")
        sys.exit(1)

    # Check for --demo flag
    run_demo = "--demo" in sys.argv

    # Initialize pipeline
    from models.pipeline import AgriMultimodalPipeline
    pipeline = AgriMultimodalPipeline()

    if run_demo:
        demo_queries(pipeline)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
