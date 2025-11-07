# Changelog

## [0.1.7] - 2025-11-07

### Changed
- Bumped package version to prepare the next PyPI release triggered from GitHub
  releases.
- Updated the automated PyPI workflow to validate and publish build artifacts
  during each release event.

## [0.1.5] - 2025-07-25

### Fixed
- **Audio Format Compatibility**: Fixed ValueError in TransformersBackend for various audio formats
  - Added robust audio preprocessing with librosa before pipeline processing
  - Now supports OPUS, OGG, M4A and other formats that transformers pipeline couldn't handle directly
  - Better error messages for unsupported or corrupted audio files
  - Added audio validation to prevent empty file processing

### Technical Details
- Audio files are now preprocessed with librosa.load() before being passed to the transformers pipeline
- Enhanced error handling with specific guidance on supported formats
- Prevents pipeline failures from ffmpeg_read issues

## [0.1.4] - 2025-07-25

### Improved
- **Enhanced Transformers Backend**: Updated TransformersBackend to use WhisperForConditionalGeneration and WhisperProcessor
  - Changed from AutoModelForSpeechSeq2Seq to WhisperForConditionalGeneration for better Whisper model compatibility
  - Improved pipeline configuration with explicit tokenizer and feature_extractor setup
  - Better device handling with torch.device() objects
  - Simplified model loading process for more reliable initialization
  - Direct audio file processing through pipeline for improved performance

### Technical Details
- Replaced AutoModelForSpeechSeq2Seq/AutoProcessor with Whisper-specific classes
- Enhanced device management and model placement
- Streamlined pipeline configuration matching tested patterns

## [0.1.3] - 2025-07-23

### Fixed
- **Word Timestamps Error**: Fixed `RuntimeError: The model configuration does not contain the field 'alignment_heads'` 
  - Added graceful fallback when word-level timestamps are not supported by the model
  - Automatically retries transcription without word timestamps when alignment heads are missing
  - Clear warning messages explaining the limitation and suggesting compatible models
  - Preserves all other functionality while disabling only the problematic word timestamps

### Improved
- Enhanced error handling for models that don't support word-level timing
- Better user guidance on model compatibility for advanced features

## [0.1.2] - 2025-07-23

### Fixed
- **Language Detection Error**: Fixed `IndexError: list index out of range` that occurred during language detection in faster_whisper
  - Added graceful fallback to English when automatic language detection fails
  - Improved error messages with clear guidance for users
  - Added comprehensive error handling for corrupted or problematic audio files
  - Added warnings to inform users about language detection issues

### Improved
- Better error messages for transcription failures
- Enhanced audio file validation with clearer error descriptions

## [0.1.1] - 2025-07-22

### Fixed
- **Import Error**: Fixed `ImportError: cannot import name 'FasterWhisperBackend'` 
  - Corrected backend class names in `__init__.py`:
    - `FasterWhisperBackend` → `CT2Backend`
    - Removed non-existent `OpenAIWhisperBackend`
  - Updated documentation examples to use correct backend names

### Changed
- Updated documentation to reflect correct backend usage (`backend="ct2"` instead of `backend="faster-whisper"`)

## [0.1.0] - 2025-07-22

### Added
- Initial release of pingala-shunya package
- Support for CT2Backend (faster-whisper) and TransformersBackend
- Unified API for speech transcription
- Word-level timestamps and confidence scores
- Auto-detection of optimal backend for each model
- CLI interface
- Comprehensive transcription features 