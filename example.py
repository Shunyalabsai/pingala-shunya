#!/usr/bin/env python3
"""
Comprehensive example script for Pingala Shunya multi-backend transcription by Shunya Labs.

This script demonstrates usage of both backends:
- ct2: High-performance CTranslate2 optimization
- transformers: Hugging Face models and latest research

Usage:
    python example.py audio.wav [model_name] [--backend backend_name]

Examples:
    python example.py audio.wav                           # Auto-detect backend  
    python example.py audio.wav --backend ct2             # Use ct2
    python example.py audio.wav --backend transformers   # Use transformers
    python example.py audio.wav shunyalabs/pingala-v1-en-verbatim  # Specific model
    python example.py audio.wav distil-whisper/distil-large-v2 # HF model

Supported models (examples):
- Default: shunyalabs/pingala-v1-en-verbatim
- Shunya Labs: shunyalabs/pingala-v1-en-verbatim
- Hugging Face: distil-whisper/distil-large-v2, microsoft/speecht5_asr
- Local: /path/to/local/model

Developed by Shunya Labs for superior transcription quality.
"""

import sys
import argparse
from pathlib import Path

from pingala_shunya import PingalaTranscriber


def backend_comparison_example(audio_path: str):
    """Compare both backends with the same model."""
    print("Backend Comparison Example")
    print("=" * 50)
    
    # Test model that works across backends
    test_model = "shunyalabs/pingala-v1-en-verbatim"
    
    backends = ["ct2", "transformers"]
    results = {}
    
    for backend_name in backends:
        print(f"\nTesting {backend_name} backend...")
        try:
            transcriber = PingalaTranscriber(
                model_name=test_model,
                backend=backend_name,
                device="cuda"
            )
            
            # Get model info
            model_info = transcriber.get_model_info()
            print(f"  Model loaded: {model_info['model_name']} on {model_info['device']}")
            
            # Simple transcription
            segments = transcriber.transcribe_file_simple(audio_path)
            results[backend_name] = {
                'segments': len(segments),
                'text': segments[0].text[:100] + "..." if segments else "No transcription",
                'info': model_info
            }
            
            print(f"  Segments found: {len(segments)}")
            if segments:
                print(f"  First segment: {segments[0].text[:100]}...")
                
        except Exception as e:
            print(f"  FAILED: {e}")
            results[backend_name] = {'error': str(e)}
    
    # Summary
    print(f"\nBackend Comparison Summary:")
    for backend, result in results.items():
        if 'error' in result:
            print(f"  {backend}: FAILED {result['error']}")
        else:
            print(f"  {backend}: SUCCESS {result['segments']} segments")


def basic_transcription_example(audio_path: str, model_name: str = None, backend: str = None):
    """Basic transcription with auto-detection or specified backend."""
    print("Basic Transcription Example")
    print("=" * 50)
    
    print(f"Audio file: {audio_path}")
    print(f"Model: {model_name or 'default (shunyalabs/pingala-v1-en-verbatim)'}")
    print(f"Backend: {backend or 'auto-detect'}")
    
    try:
        # Initialize transcriber
        transcriber = PingalaTranscriber(
            model_name=model_name,
            backend=backend,
            device="cuda"
        )
        
        # Get model info
        model_info = transcriber.get_model_info()
        print(f"\nModel Info:")
        print(f"  Backend: {model_info['backend']}")
        print(f"  Model: {model_info['model_name']}")
        print(f"  Device: {model_info['device']}")
        
        # Simple transcription
        print(f"\nTranscribing...")
        segments = transcriber.transcribe_file_simple(audio_path, beam_size=5)
        
        print(f"\nTranscription Results ({len(segments)} segments):")
        for segment in segments:
            print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {segment.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")
        if "not installed" in str(e):
            print("HINT: Install the required backend dependencies:")
            print("   pip install 'pingala-shunya[transformers]'  # for transformers")
            print("   pip install 'pingala-shunya'  # for ct2 (default)")


def advanced_transcription_example(audio_path: str, model_name: str = None, backend: str = None):
    """Advanced transcription with full metadata and features."""
    print("Advanced Transcription Example")
    print("=" * 50)
    
    try:
        transcriber = PingalaTranscriber(
            model_name=model_name or "shunyalabs/pingala-v1-en-verbatim",
            backend=backend,
            device="cuda", 
            compute_type="float16"
        )
        
        model_info = transcriber.get_model_info()
        print(f"Backend: {model_info['backend']}, Model: {model_info['model_name']}")
        
        # Advanced transcription with all parameters
        print(f"\nAdvanced transcription with full parameters...")
        segments, info = transcriber.transcribe_file(
            audio_path,
            beam_size=10,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            initial_prompt="High quality audio recording",
            task="transcribe",
            language="en"
        )
        
        # Display info
        print(f"\nTranscription Info:")
        print(f"  Language: {info.language} (confidence: {info.language_probability:.3f})")
        print(f"  Duration: {info.duration:.2f} seconds")
        print(f"  Duration after VAD: {info.duration_after_vad:.2f} seconds")
        
        # Display segments with metadata
        print(f"\nDetailed Segments ({len(segments)} total):")
        for i, segment in enumerate(segments[:3], 1):  # Show first 3
            print(f"\nSegment {i}:")
            print(f"  Time: [{segment.start:.2f}s -> {segment.end:.2f}s]")
            print(f"  Text: {segment.text}")
            if segment.confidence:
                print(f"  Confidence: {segment.confidence:.3f}")
            if segment.avg_logprob:
                print(f"  Avg Log Prob: {segment.avg_logprob:.3f}")
            if segment.no_speech_prob:
                print(f"  No Speech Prob: {segment.no_speech_prob:.3f}")
                
        if len(segments) > 3:
            print(f"\n... and {len(segments) - 3} more segments")
            
    except Exception as e:
        print(f"ERROR: {e}")


def word_level_timestamps_example(audio_path: str, model_name: str = None, backend: str = None):
    """Demonstrate word-level timestamps (ct2 only)."""
    print("Word-Level Timestamps Example")
    print("=" * 50)
    
    # Choose backend that supports word timestamps
    if backend == "transformers":
        print("WARNING: Word-level timestamps have limited support in transformers backend.")
        print("    Using ct2 for this example.")
        backend = "ct2"
    
    try:
        transcriber = PingalaTranscriber(
            model_name=model_name or "shunyalabs/pingala-v1-en-verbatim",
            backend=backend or "ct2",
            device="cuda"
        )
        
        model_info = transcriber.get_model_info()
        print(f"Backend: {model_info['backend']}, Model: {model_info['model_name']}")
        
        print(f"\nTranscribing with word-level timestamps...")
        segments, info = transcriber.transcribe_with_word_timestamps(
            audio_path,
            beam_size=5,
            language="en"
        )
        
        print(f"\nWord-Level Results ({len(segments)} segments):")
        for i, segment in enumerate(segments[:2], 1):  # Show first 2 segments
            print(f"\nSegment {i}: [{segment.start:.2f}s -> {segment.end:.2f}s]")
            print(f"Text: {segment.text}")
            
            if segment.words:
                print(f"Words ({len(segment.words)}):")
                for word in segment.words:
                    print(f"  '{word.word}' [{word.start:.2f}-{word.end:.2f}s] (conf: {word.probability:.3f})")
            else:
                print("  No word-level data available")
                
        if len(segments) > 2:
            print(f"\n... and {len(segments) - 2} more segments")
            
    except Exception as e:
        print(f"ERROR: {e}")


def vad_filtering_example(audio_path: str, model_name: str = None):
    """Demonstrate Voice Activity Detection (ct2 only)."""
    print("Voice Activity Detection (VAD) Example")
    print("=" * 50)
    
    try:
        transcriber = PingalaTranscriber(
            model_name=model_name or "shunyalabs/pingala-v1-en-verbatim",
            backend="ct2",  # VAD only available in ct2
            device="cuda"
        )
        
        print("Backend: ct2 (VAD filtering)")
        
        # VAD parameters
        vad_params = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float("inf"),
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400
        }
        
        print(f"\nTranscribing with VAD filtering...")
        print(f"VAD params: {vad_params}")
        
        segments, info = transcriber.transcribe_with_vad(
            audio_path,
            vad_parameters=vad_params,
            beam_size=5
        )
        
        print(f"\nVAD Results:")
        print(f"  Original duration: {info.duration:.2f}s")
        print(f"  After VAD filtering: {info.duration_after_vad:.2f}s")
        print(f"  Time saved: {info.duration - info.duration_after_vad:.2f}s")
        print(f"  Segments found: {len(segments)}")
        
        print(f"\nVAD-Filtered Segments:")
        for segment in segments[:3]:  # Show first 3
            print(f"[{segment.start:6.2f}s -> {segment.end:6.2f}s] {segment.text}")
            
    except Exception as e:
        print(f"ERROR: {e}")


def language_detection_example(audio_path: str, backend: str = None):
    """Demonstrate language detection across backends."""
    print("Language Detection Example")
    print("=" * 50)
    
    backends_to_test = [backend] if backend else ["ct2", "transformers"]
    
    for backend_name in backends_to_test:
        print(f"\nTesting language detection with {backend_name}...")
        try:
            transcriber = PingalaTranscriber(
                model_name="shunyalabs/pingala-v1-en-verbatim",  # Use Shunya Labs model for detection
                backend=backend_name,
                device="cuda"
            )
            
            info = transcriber.detect_language(audio_path)
            
            print(f"  Backend: {backend_name}")
            print(f"  Detected language: {info.language}")
            print(f"  Confidence: {info.language_probability:.3f}")
            print(f"  Audio duration: {info.duration:.2f}s")
            
        except Exception as e:
            print(f"  FAILED with {backend_name}: {e}")


def streaming_transcription_example(audio_path: str, model_name: str = None, backend: str = None):
    """Demonstrate streaming transcription."""
    print("Streaming Transcription Example")
    print("=" * 50)
    
    try:
        transcriber = PingalaTranscriber(
            model_name=model_name or "shunyalabs/pingala-v1-en-verbatim",
            backend=backend or "ct2",
            device="cuda"
        )
        
        model_info = transcriber.get_model_info()
        print(f"Backend: {model_info['backend']}")
        
        if model_info['backend'] == "ct2":
            print("True streaming transcription (ct2)")
        else:
            print("Batch processing with generator interface")
        
        print(f"\nProcessing segments as they arrive...")
        segment_count = 0
        
        for segment in transcriber.transcribe_file_generator(
            audio_path,
            beam_size=5,
            word_timestamps=False
        ):
            segment_count += 1
            print(f"Segment {segment_count}: [{segment.start:6.2f}s -> {segment.end:6.2f}s] {segment.text}")
            
            # Show only first 5 segments for demo
            if segment_count >= 5:
                break
        
        print(f"... (streaming continues for full audio)")
        
    except Exception as e:
        print(f"ERROR: {e}")


def transformers_models_example(audio_path: str):
    """Demonstrate different transformer models."""
    print("Hugging Face Transformers Models Example")
    print("=" * 50)
    
    # Different transformer models to test
    models = [
        ("distil-whisper/distil-large-v2", "Distilled large-v2 (6x faster)"),
        ("distil-whisper/distil-medium.en", "English-only distilled medium"),
        ("microsoft/speecht5_asr", "Microsoft's speech recognition model"),
    ]
    
    for model_name, description in models:
        print(f"\nTesting {model_name}")
        print(f"Description: {description}")
        
        try:
            transcriber = PingalaTranscriber(
                model_name=model_name,
                backend="transformers",  # Force transformers backend
                device="cuda"
            )
            
            segments = transcriber.transcribe_file_simple(audio_path)
            
            print(f"SUCCESS: {len(segments)} segments")
            if segments:
                print(f"Sample: {segments[0].text[:80]}...")
                
        except Exception as e:
            print(f"FAILED: {e}")
            if "not installed" in str(e):
                print("HINT: Install with: pip install 'pingala-shunya[transformers]'")


def model_info_example(model_name: str = None, backend: str = None):
    """Display detailed model information."""
    print("Model Information Example")
    print("=" * 50)
    
    try:
        transcriber = PingalaTranscriber(
            model_name=model_name,
            backend=backend,
            device="cuda"
        )
        
        model_info = transcriber.get_model_info()
        
        print(f"Model Details:")
        for key, value in model_info.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Pingala Shunya examples with multi-backend support by Shunya Labs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python example.py audio.wav                           # All examples with auto-detection
  python example.py audio.wav --backend ct2             # Use ct2 
  python example.py audio.wav --backend transformers   # Use transformers
  python example.py audio.wav shunyalabs/pingala-v1-en-verbatim  # With specific model
  python example.py audio.wav distil-whisper/distil-large-v2  # HF model

Supported backends:
  - ct2: High-performance CTranslate2 optimization (default)
  - transformers: Latest models, research, Hugging Face

Developed by Shunya Labs for superior transcription quality.
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to audio file to transcribe"
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        help="Model name (default: shunyalabs/pingala-v1-en-verbatim)"
    )
    
    parser.add_argument(
        "--backend",
        choices=["ct2", "transformers"],
        help="Backend to use (auto-detects if not specified)"
    )
    
    parser.add_argument(
        "--example",
        choices=[
            "all", "basic", "advanced", "words", "vad", "language", 
            "streaming", "transformers", "compare", "info"
        ],
        default="all",
        help="Which example to run (default: all)"
    )
    
    args = parser.parse_args()
    
    # Check if audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"ERROR: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    audio_path = str(audio_path)
    
    print("Pingala Shunya Multi-Backend Examples by Shunya Labs")
    print("=" * 60)
    print(f"Audio file: {audio_path}")
    print(f"Model: {args.model_name or 'default'}")
    print(f"Backend: {args.backend or 'auto-detect'}")
    print(f"Examples: {args.example}")
    print("=" * 60)
    
    # Run selected examples
    try:
        if args.example in ["all", "info"]:
            model_info_example(args.model_name, args.backend)
            print()
        
        if args.example in ["all", "compare"]:
            backend_comparison_example(audio_path)
            print()
        
        if args.example in ["all", "basic"]:
            basic_transcription_example(audio_path, args.model_name, args.backend)
            print()
        
        if args.example in ["all", "advanced"]:
            advanced_transcription_example(audio_path, args.model_name, args.backend)
            print()
        
        if args.example in ["all", "words"]:
            word_level_timestamps_example(audio_path, args.model_name, args.backend)
            print()
        
        if args.example in ["all", "vad"]:
            vad_filtering_example(audio_path, args.model_name)
            print()
        
        if args.example in ["all", "language"]:
            language_detection_example(audio_path, args.backend)
            print()
        
        if args.example in ["all", "streaming"]:
            streaming_transcription_example(audio_path, args.model_name, args.backend)
            print()
        
        if args.example in ["all", "transformers"]:
            transformers_models_example(audio_path)
            print()
        
        print("Examples completed successfully!")
        print("\nTips:")
        print("  - Use ct2 for production and real-time applications")
        print("  - Use transformers for latest research models and fine-tuning")
        print("  - The default shunyalabs/pingala-v1-en-verbatim model is optimized for quality")
        print("  - Install extras: pip install 'pingala-shunya[all]' for all backends")
        print("  - Developed by Shunya Labs for superior transcription quality")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 