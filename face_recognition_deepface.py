# face_recognition_cosine.py
import os
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pickle

# CONFIG
DATASET_DIR = "dataset"
MODEL_DIR = "models"
MODEL_NAME = "ArcFace"
SIMILARITY_THRESHOLD = 0.35  # lower -> more sensitive, higher -> stricter
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------------------------------
# STEP 1 — Build face embeddings database
# -------------------------------------------------------
def build_embeddings_db():
    print(f"\nBuilding embeddings database using {MODEL_NAME}...")
    db = {}
    for person in sorted(os.listdir(DATASET_DIR)):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        embeddings = []
        for fname in os.listdir(person_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(person_path, fname)
            try:
                reps = DeepFace.represent(
                    img_path=img_path,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )
                if reps and isinstance(reps, list) and "embedding" in reps[0]:
                    embeddings.append(reps[0]["embedding"])
            except Exception as e:
                print("Skipped:", img_path, "-", e)

        if len(embeddings) > 0:
            db[person] = np.mean(embeddings, axis=0)
            print(f"Stored embedding for {person} ({len(embeddings)} images)")
    with open(os.path.join(MODEL_DIR, "embeddings_db.pkl"), "wb") as f:
        pickle.dump(db, f)
    print(f"Saved embeddings database to '{MODEL_DIR}/embeddings_db.pkl'")


# -------------------------------------------------------
# STEP 2 — Live face recognition via cosine similarity
# -------------------------------------------------------
def live_recognition():
    if not os.path.exists(os.path.join(MODEL_DIR, "embeddings_db.pkl")):
        raise SystemExit("No database found. Run training first (option 1).")

    with open(os.path.join(MODEL_DIR, "embeddings_db.pkl"), "rb") as f:
        db = pickle.load(f)

    print("\nStarting live recognition... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Cannot access webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            detections = DeepFace.represent(
                rgb_frame, model_name=MODEL_NAME, enforce_detection=False
            )
        except Exception:
            detections = []

        for det in detections:
            if "embedding" not in det or "facial_area" not in det:
                continue

            emb = np.array(det["embedding"]).reshape(1, -1)
            facial_area = det["facial_area"]
            x, y, w, h = (
                facial_area.get("x", 0),
                facial_area.get("y", 0),
                facial_area.get("w", 0),
                facial_area.get("h", 0),
            )

            # Compare embedding with all known faces
            best_name = "Unknown"
            best_score = 0.0
            for person, known_emb in db.items():
                sim = cosine_similarity(emb, known_emb.reshape(1, -1))[0][0]
                if sim > best_score:
                    best_score = sim
                    best_name = person

            # Decision based on similarity
            if best_score < SIMILARITY_THRESHOLD:
                best_name = "Unknown"

            label = f"{best_name} ({best_score:.2f})"
            color = (0, 255, 0) if best_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------
if __name__ == "__main__":
    print("1. Build embeddings database (train)")
    print("2. Live recognition (requires trained embeddings)")
    choice = input("Choose option (1 or 2): ").strip()

    if choice == "1":
        build_embeddings_db()
    elif choice == "2":
        live_recognition()
    else:
        print("Invalid choice.")
