from pathlib import Path

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from langchain.docstore.document import Document
from src.utils.conversation import saveConversation

from src.libs.messages import (print_info_message, print_success_message,
                               print_error_message, print_plugin_message)


_processor = None
_model = None

if torch.cuda.is_available():
    DEVICE = "cuda"
    print_info_message("Using GPU for VLM processing.")
else:
    DEVICE = "cpu"
    print_info_message("No GPU found, using CPU for VLM processing. This may be slow.")


def _ingest_conversation_turn(user_input, aeon_output, vectorstore, text_splitter, llama_embeddings):
    try:
        conversation_text = f"{user_input}\n\n{aeon_output}"
        
        conversation_document = Document(
            page_content=conversation_text,
            metadata={"source": "smolvlm-256m-instruct"}
        )
        
        docs = text_splitter.split_documents([conversation_document])
        success, failed = 0, 0
        for i, chunk in enumerate(docs, start=1):
            try:
                vectorstore.add_documents([chunk])
                success += 1
            except Exception as e:
                failed += 1
                print_error_message(f" Failed on chunk {i}: {e}")
    except Exception as e:
        print_error_message(f"Failed to ingest conversation turn: {e}")


def get_pipeline(plugin_config: dict, plugin_dir: Path):
    global _processor, _model

    if _processor is None or _model is None:
        if not plugin_config:
            print_error_message("VLM pipeline cannot be initialized; plugin configuration not provided.")
            return None, None

        model_path_str = plugin_config.get("model_path")
        
        if not model_path_str:
            print_error_message("Model ID or path is missing in plugin configuration.")
            return None, None
            
        full_model_path = plugin_dir / model_path_str

        print_plugin_message(f"Initializing VLM pipeline for {model_path_str}...")
        try:
            _processor = AutoProcessor.from_pretrained(
                full_model_path, local_files_only=True, cache_dir=full_model_path.parent
            )
            _model = AutoModelForVision2Seq.from_pretrained(
                full_model_path, local_files_only=True, cache_dir=full_model_path.parent
            )
            _model.to(DEVICE)
            print_plugin_message("VLM model and processor loaded successfully.")
        except Exception as e:
            print_error_message(f"Failed to load VLM model: {e}")
            _processor, _model = None, None  # Reset on failure

    return _processor, _model


def run_plugin(args: str, **kwargs) -> str:
    plugin_config = kwargs.get('plugin_config')
    plugin_dir = kwargs.get('plugin_dir')
    plugin_name = plugin_config.get("plugin_name")
    vectorstore = kwargs.get('vectorstore')
    text_splitter = kwargs.get('text_splitter')
    llama_embeddings = kwargs.get('llama_embeddings')
    conversation_filename = kwargs.get('conversation_filename')
    current_memory_path = kwargs.get('current_memory_path')
    current_chat_history=kwargs.get("current_chat_history")


    if not args:
        print_error_message("An image path and prompt are required for VLM processing.")
        return {"success": False, "message": "An image path and prompt are required."}
    
    # Split the single 'args' string into image_path and prompt
    try:
        parts = args.split(" ", 1)
        image_path = parts[0]
        prompt = parts[1] if len(parts) > 1 else ""
    except IndexError:
        print_error_message("Invalid arguments format. Please provide an image path and a prompt.")
        return {"success": False, "message": "Invalid arguments format."}

    processor, model = get_pipeline(plugin_config, plugin_dir)
    if not processor or not model:
        print_error_message("VLM pipeline is not available. Check plugin configuration and model files.")
        return {"success": False, "message": "VLM pipeline not available."}

    image_file = Path(image_path)
    if not image_file.exists():
        print_error_message(f"Image file not found at: {image_file}")
        return {"success": False, "message": "Image file not found."}

    print_plugin_message(f"Processing image '{image_file.name}' with prompt: '{prompt}'...")
    try:
        image = Image.open(image_file).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"{prompt}"}
                ]
            },
        ]
        
        input_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        inputs = processor(
            text=input_prompt,
            images=image,
            return_tensors="pt"
        ).to(DEVICE)

        generated_ids = model.generate(**inputs, max_new_tokens=256)
        vlm_response_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # This is the core fix: use the VLM's output as the input for the RAG chain.
        print_plugin_message(f"VLM Response: {vlm_response_text}")

        rag_chain = kwargs.get("rag_chain")
        if not rag_chain:
            print_error_message("RAG system not initialized.")
            return {"success": False, "message": "RAG system not initialized."}

        try:
            print_plugin_message("Querying RAG system with VLM response...")
            result = rag_chain.invoke(vlm_response_text)
            aeon_response_text = result.get("answer", "No answer found.")

            if not aeon_response_text:
                print_plugin_message("RAG system returned an empty response. Cannot ingest conversation.")
                return {"success": False, "message": "RAG system returned an empty response."}

            full_user_input = f"VLM Analysis of '{image_file.name}' based on prompt: '{prompt}'"
            full_aeon_output = f"VLM Response: {vlm_response_text}\n\nAeon RAG Response: {aeon_response_text}"

            _ingest_conversation_turn(
                user_input=full_user_input,
                aeon_output=full_aeon_output,
                vectorstore=vectorstore,
                text_splitter=text_splitter,
                llama_embeddings=llama_embeddings
            )

            saveConversation(
                full_user_input,
                full_aeon_output,
                plugin_name,
                current_memory_path,
                conversation_filename
            )

            print_plugin_message(f"[AEON]: {aeon_response_text}")
            
            return {"success": True, "message": f"Image successfully processed. RAG Response: {aeon_response_text}"}

        except Exception as e:
            error_msg = f"An error occurred during RAG chain processing: {type(e).__name__}: {e}"
            print_error_message(error_msg)
            return {"success": False, "message": error_msg}

    except Exception as e:
        error_msg = f"An error occurred during VLM processing: {type(e).__name__}: {e}"
        print_error_message(error_msg)
        return {"success": False, "message": error_msg}
