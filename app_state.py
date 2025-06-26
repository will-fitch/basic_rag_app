from rag_model import RAGModel, RAGResponse
import file_storage

class AppState:
    """
    Manages the application state and business logic for the RAG chatbot.
    Separates business logic from UI concerns.
    """
    
    def __init__(self):
        self.rag_model: RAGModel = RAGModel()
        for filename in self.list_uploaded_files():
            file_text = self.load_file_content(filename)
            self.rag_model.add_content(file_text, filename)
        
    def query_model(self, query: str) -> RAGResponse:
        """Query the RAG model and return response with sources."""
        return self.rag_model.query(query)

    # File storage logic
    def load_file_content(self, filename: str) -> str:
        return file_storage.load_file_content(filename)

    def list_uploaded_files(self):
        return file_storage.list_uploaded_files()

    def delete_file(self, filename: str) -> None:
        file_storage.delete_uploaded_file(filename)
        self.rag_model.delete_by_filename(filename)

    def upload_file(self, file_name: str, file_content: bytes, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        """Save a file and add its content to the RAG model."""

        if file_storage.filename_exists(file_name):
            raise RuntimeError("File with that name already exists")

        # Save file to disk
        file_storage.save_uploaded_file(file_name, file_content)
        # Load file content as text
        ext = file_storage.get_file_extension(file_name)
        if ext == ".pdf":
            file_text = file_storage.extract_text_from_pdf(file_content)
        else:
            file_text = file_storage.load_file_content(file_name)
        # Add to RAG model
        return self.rag_model.add_content(file_text, file_name, chunk_size, chunk_overlap)