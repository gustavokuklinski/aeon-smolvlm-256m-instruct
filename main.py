import subprocess
import tempfile
import threading
import time
from pathlib import Path
import shutil
import cv2
import sys
import numpy as np
from pynput import keyboard

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from langchain.docstore.document import Document
from src.utils.conversation import saveConversation
from src.libs.messages import (print_info_message, print_success_message,
                               print_error_message, print_plugin_message)

_processor = None
_model = None

# Events for keyboard control
_stop_event = threading.Event()
_capture_event = threading.Event()

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
            _processor, _model = None, None

    return _processor, _model


def _process_image_with_vlm(image: Image.Image, prompt: str, plugin_config: dict, plugin_dir: Path, kwargs: dict):
    """
    Processes a PIL Image with the VLM model.
    """
    processor, model = get_pipeline(plugin_config, plugin_dir)
    if not processor or not model:
        return {"success": False, "message": "VLM pipeline not available."}

    try:
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

        return {"success": True, "message": vlm_response_text}

    except Exception as e:
        error_msg = f"An error occurred during VLM processing: {type(e).__name__}: {e}"
        print_error_message(error_msg)
        return {"success": False, "message": error_msg}


def _process_and_ingest(image_source, prompt, plugin_config, plugin_dir, kwargs, is_camera=False):
    """
    Centralized function to process an image with VLM and then ingest the response into RAG.
    """
    try:
        if isinstance(image_source, Path):
            image = Image.open(image_source).convert("RGB")
        else:
            image = image_source

        vlm_result = _process_image_with_vlm(image, prompt, plugin_config, plugin_dir, kwargs)

        if not vlm_result["success"]:
            return vlm_result

        vlm_response_text = vlm_result["message"]
        print_plugin_message(f"[VLM]: {vlm_response_text}")

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

            if is_camera:
                full_user_input = f"{prompt}"
            else:
                full_user_input = f"{image_source.name}\n\n{prompt}'"

            full_aeon_output = f"{vlm_response_text}\n\n{aeon_response_text}"

            _ingest_conversation_turn(
                user_input=full_user_input,
                aeon_output=full_aeon_output,
                vectorstore=kwargs.get("vectorstore"),
                text_splitter=kwargs.get("text_splitter"),
                llama_embeddings=kwargs.get("llama_embeddings")
            )

            saveConversation(
                full_user_input,
                full_aeon_output,
                plugin_config.get("plugin_name"),
                kwargs.get("current_memory_path"),
                kwargs.get("conversation_filename")
            )

            print_plugin_message(f"[AEON]: {aeon_response_text}")
            print_info_message("If on live detection: Press 'Q' to stop the live detection or 'SPACE' to take a picture and analyze it.")
            return {"success": True, "message": f"Image successfully processed. RAG Response: {aeon_response_text}"}

        except Exception as e:
            error_msg = f"An error occurred during RAG chain processing: {type(e).__name__}: {e}"
            print_error_message(error_msg)
            return {"success": False, "message": error_msg}

    except Exception as e:
        error_msg = f"An error occurred during VLM processing: {type(e).__name__}: {e}"
        print_error_message(error_msg)
        return {"success": False, "message": error_msg}


def _on_press(key):
    try:
        if key == keyboard.Key.space:
            _capture_event.set()
        elif key.char == 'q':
            _stop_event.set()
    except AttributeError:
        # Ignore non-character keys
        pass


def _live_object_detection(prompt, plugin_config, plugin_dir, kwargs):
    print_info_message("Accessing camera.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print_error_message("Could not open video device. It may be in use by another application or you may not have permissions.")
        return {"success": False, "message": "Could not open video device."}

    print_info_message("Starting live object detection.")
    print_info_message("Press 'Q' to stop the live detection or 'SPACE' to take a picture and analyze it.")

    listener = keyboard.Listener(on_press=_on_press)
    listener.start()

    try:
        while not _stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print_error_message("Failed to read frame from camera. Stopping.")
                break

            if _capture_event.is_set():
                print_plugin_message("Capturing and processing frame...")
                # Convert the NumPy array (BGR) to a PIL Image (RGB) for the model
                pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = _process_and_ingest(pil_image, prompt, plugin_config, plugin_dir, kwargs, is_camera=True)

                if not result["success"]:
                    print_error_message(f"Processing failed: {result['message']}")
                
                # Reset the event to wait for the next key press
                _capture_event.clear()

            time.sleep(0.1)  # Small delay to avoid 100% CPU usage

    except Exception as e:
        print_error_message(f"An unexpected error occurred: {e}")
    finally:
        cap.release()
        listener.stop()
        print_info_message("Camera released and keyboard listener stopped.")

    return {"success": True, "message": "Live object detection session ended."}


def run_plugin(args: str, **kwargs) -> str:
    plugin_config = kwargs.get('plugin_config')
    plugin_dir = kwargs.get('plugin_dir')

    if not args:
        print_error_message("An image path and prompt are required for VLM processing.")
        return {"success": False, "message": "An image path and prompt are required."}

    # Check for the live camera command
    if args.startswith("/camera"):
        prompt = args[len("/camera"):].strip()
        if not prompt:
            print_error_message("Please provide a prompt after '/camera'.")
            return {"success": False, "message": "Please provide a prompt after '/camera'."}
        return _live_object_detection(prompt, plugin_config, plugin_dir, kwargs)
    else:
        # Assume it's a static image command
        try:
            parts = args.split(" ", 1)
            image_path = parts[0]
            prompt = parts[1] if len(parts) > 1 else ""
        except IndexError:
            print_error_message("Invalid arguments format. Please provide an image path and a prompt.")
            return {"success": False, "message": "Invalid arguments format."}

        image_file = Path(image_path)
        if not image_file.exists():
            print_error_message(f"Image file not found at: {image_file}")
            return {"success": False, "message": "Image file not found."}

        print_plugin_message(f"Processing image '{image_file.name}' with prompt: '{prompt}'...")
        return _process_and_ingest(image_file, prompt, plugin_config, plugin_dir, kwargs)
