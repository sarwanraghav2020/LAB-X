import unittest
import numpy as np
import io
from PIL import Image
from utils import Document, MultimodalPredictor, MedicalLiteratureDB, DocumentSplitter, MockVectorDB, ClinicalRAG

class TestLABXBackend(unittest.TestCase):
    
    def setUp(self):
        # Initialize components
        self.predictor = MultimodalPredictor()
        self.lit_db = MedicalLiteratureDB()
        
        # Load raw articles
        articles = self.lit_db.get_articles()
        raw_docs = []
        for art in articles:
            content = f"Document ID: {art['id']}\nTitle: {art['title']}\nAuthors: {art['authors']}\nJournal: {art['journal']} ({art['year']})\nTopic: {art['topic']}\n\nAbstract Details:\n{art['content']}"
            metadata = {
                "id": art["id"],
                "title": art["title"],
                "authors": art["authors"],
                "journal": art["journal"],
                "year": art["year"],
                "topic": art["topic"],
                "source": f"PubMed://{art['id']}"
            }
            raw_docs.append(Document(page_content=content, metadata=metadata))
            
        # Split into optimized chunks
        self.splitter = DocumentSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = self.splitter.split(raw_docs)
        self.vector_db = MockVectorDB(self.chunks)
        self.rag = ClinicalRAG(self.vector_db)
        
        # Create a dummy test image and save it as a BytesIO file-like object
        img_arr = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        dummy_image = Image.fromarray(img_arr)
        self.dummy_image_file = io.BytesIO()
        dummy_image.save(self.dummy_image_file, format='JPEG')
        self.dummy_image_file.seek(0)
        
    def test_predictor_modalities(self):
        """Test classification and Grad-CAM coordinate mapping for all modalities."""
        modalities = ["Chest X-Ray", "Brain MRI", "Abdominal CT", "Knee MRI"]
        for modality in modalities:
            # Seek back to 0 for consecutive reads
            self.dummy_image_file.seek(0)
            condition, probs, blended_img, obs = self.predictor.predict(
                self.dummy_image_file, 
                modality=modality, 
                scenario="Auto-Detect"
            )
            # Verify predictions exist
            self.assertIsNotNone(condition)
            self.assertIn(condition, probs)
            self.assertEqual(probs[condition], max(probs.values()))
            
            # Verify Grad-CAM output shape and size (standardized to 512x512)
            self.assertEqual(blended_img.size, (512, 512))
            
    def test_literature_db(self):
        """Test that literature database loaded articles correctly."""
        articles = self.lit_db.get_articles()
        self.assertGreater(len(articles), 0)
        for article in articles:
            self.assertIn("id", article)
            self.assertIn("title", article)
            self.assertIn("content", article)
                
    def test_vector_db_search(self):
        """Test TF-IDF semantic query matching in the Mock FAISS database."""
        query = "pneumonia consolidation antibiotics"
        results = self.vector_db.similarity_search(query, k=2)
        self.assertEqual(len(results), 2)
        
    def test_rag_synthesis(self):
        """Test simulated LLM response formatting."""
        diagnosis = "Pneumonia"
        observations = "Infiltrates in right lower lung"
        summary = self.rag.summarize_diagnosis(diagnosis, observations)
        
        self.assertIn("Executive Diagnostic Summary", summary)
        self.assertIn("Pneumonia", summary)

if __name__ == "__main__":
    print("🚀 Starting LAB X Backend Verification Suite...")
    unittest.main()
