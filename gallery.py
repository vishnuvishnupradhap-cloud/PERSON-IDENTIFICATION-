import pickle
import os
import numpy as np

class ReIDGallery:
    def __init__(self, similarity_threshold=0.75, max_history=10):
        """
        Initializes the Person Re-ID Gallery.
        :param similarity_threshold: Minimum cosine similarity to match an identity.
        :param max_history: Maximum number of embeddings to keep per identity.
        """
        self.similarity_threshold = similarity_threshold
        self.max_history = max_history
        self.identities = {}  # Format: {id_str: {"embeddings": [emb1, emb2, ...], "colors": {"upper": "...", "lower": "..."}, "thumbnail": np.ndarray}}
        self.next_id_num = 1

    def match_or_create(self, embedding, colors, thumbnail=None):
        """
        Matches a query embedding against the gallery. If matched, updates the identity.
        If not matched, creates a new unique identity.
        :param embedding: L2-normalized 512-dim numpy array.
        :param colors: dict with "upper" and "lower" clothing colors.
        :param thumbnail: cropped BGR image of the person (optional).
        :return: assigned identity string, maximum similarity score, and whether it is a new person.
        """
        if len(self.identities) == 0:
            # First person detected, create new identity
            new_id = self._create_new_identity(embedding, colors, thumbnail)
            return new_id, 1.0, True

        best_id = None
        best_sim = -1.0

        # Compare against all stored embeddings of all identities
        for id_str, id_data in self.identities.items():
            stored_embs = id_data["embeddings"]
            
            # Compute dot product (which is cosine similarity since they are L2-normalized)
            # stored_embs is a list of arrays of shape (512,)
            # We can stack them and compute dot product in batch
            embs_array = np.vstack(stored_embs)
            similarities = np.dot(embs_array, embedding)
            
            # Find the max similarity for this identity (could also use average of top-k)
            max_sim = np.max(similarities)
            
            if max_sim > best_sim:
                best_sim = max_sim
                best_id = id_str

        # If similarity is above threshold, match to existing identity
        if best_sim >= self.similarity_threshold:
            # Update matching identity's list of embeddings
            id_data = self.identities[best_id]
            id_data["embeddings"].append(embedding)
            if len(id_data["embeddings"]) > self.max_history:
                id_data["embeddings"].pop(0)  # Remove oldest to avoid memory leak and drift
            
            # Update clothing colors (majority vote or latest)
            id_data["colors"] = colors
            
            # Update thumbnail to latest high-quality one
            if thumbnail is not None:
                id_data["thumbnail"] = thumbnail
                
            return best_id, float(best_sim), False
        else:
            # Create new identity
            new_id = self._create_new_identity(embedding, colors, thumbnail)
            return new_id, float(best_sim), True

    def _create_new_identity(self, embedding, colors, thumbnail=None):
        """
        Registers a new unique identity.
        """
        id_str = f"Person_{self.next_id_num}"
        self.next_id_num += 1
        
        self.identities[id_str] = {
            "embeddings": [embedding],
            "colors": colors,
            "thumbnail": thumbnail
        }
        print(f"[Gallery] Registered new identity: {id_str} (Upper: {colors['upper']}, Lower: {colors['lower']})")
        return id_str

    def save_gallery(self, filepath="gallery.pkl"):
        """
        Saves the current gallery state to disk.
        """
        with open(filepath, "wb") as f:
            pickle.dump({
                "identities": self.identities,
                "next_id_num": self.next_id_num,
                "similarity_threshold": self.similarity_threshold,
                "max_history": self.max_history
            }, f)
        print(f"[Gallery] Saved gallery database containing {len(self.identities)} people to {filepath}")

    def load_gallery(self, filepath="gallery.pkl"):
        """
        Loads the gallery state from disk.
        """
        if not os.path.exists(filepath):
            print(f"[Gallery] No gallery database found at {filepath}. Starting empty.")
            return False
            
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.identities = data.get("identities", {})
            self.next_id_num = data.get("next_id_num", 1)
            self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
            self.max_history = data.get("max_history", self.max_history)
            
        print(f"[Gallery] Loaded gallery database containing {len(self.identities)} people from {filepath}")
        return True
