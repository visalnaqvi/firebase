import torch
from qdrant_client import QdrantClient
from collections import defaultdict

class PersonMerger:
    def __init__(self, host="localhost", port=6333, collection_name="gallary-v2"):
        self.qdrant = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def fetch_all_points(self):
        points = []
        offset = None
        while True:
            result = self.qdrant.scroll(
                collection_name=self.collection_name,
                with_vectors=True,
                with_payload=True,
                limit=1000,
                offset=offset
            )
            points.extend(result[0])
            if result[1] is None:
                break
            offset = result[1]
        return points

    def compute_centroid(self, vectors):
        """Compute average embedding"""
        tensor_stack = torch.stack(vectors)
        return tensor_stack.mean(dim=0)

    def merge_person_ids(self, face_th=0.75, cloth_th=0.85):
        print("🔍 Starting second-pass deduplication...")
        points = self.fetch_all_points()
        print(f"✅ Fetched {len(points)} points")

        # Group points by person_id
        person_groups = defaultdict(list)
        for p in points:
            pid = p.payload.get("person_id")
            face_vec = torch.tensor(p.vector["face"])
            cloth_vec = torch.tensor(p.vector["cloth"])
            person_groups[pid].append((face_vec, cloth_vec, p.id))

        # Compute centroids
        centroids = {}
        for pid, vecs in person_groups.items():
            face_vectors = [v[0] for v in vecs]
            cloth_vectors = [v[1] for v in vecs]
            centroids[pid] = {
                "face": self.compute_centroid(face_vectors),
                "cloth": self.compute_centroid(cloth_vectors)
            }

        # Compare centroids to find duplicates
        pid_list = list(centroids.keys())
        merged = {}
        for i in range(len(pid_list)):
            for j in range(i+1, len(pid_list)):
                pid_a = pid_list[i]
                pid_b = pid_list[j]
                if pid_a in merged or pid_b in merged:
                    continue  # Already merged

                face_sim = 1 - torch.nn.functional.cosine_similarity(
                    centroids[pid_a]["face"].unsqueeze(0),
                    centroids[pid_b]["face"].unsqueeze(0)
                ).item()
                cloth_sim = float(centroids[pid_a]["cloth"] @ centroids[pid_b]["cloth"])

                if face_sim >= face_th or (face_sim >= 0.5 and cloth_sim >= cloth_th):
                    print(f"✅ Merge {pid_b} → {pid_a} (Face: {face_sim:.3f}, Cloth: {cloth_sim:.3f})")
                    merged[pid_b] = pid_a

        # Apply merges
        for old_pid, new_pid in merged.items():
            ids_to_update = [p[2] for p in person_groups[old_pid]]
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"person_id": new_pid},
                points=ids_to_update
            )

        print(f"🔄 Completed merging. {len(merged)} person_ids merged.")

# ✅ Run Deduplication
if __name__ == "__main__":
    merger = PersonMerger()
    merger.merge_person_ids(face_th=0.75, cloth_th=0.85)
