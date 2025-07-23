# Changelog

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