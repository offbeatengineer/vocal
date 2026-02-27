/**
 * Vocal SDK Usage Examples
 */

import { createVocal, Vocal } from 'vocal';

// Create instance
const vocal = createVocal(process.env.MINIMAX_API_KEY || 'your-api-key');

/**
 * Example 1: Simplified voice clone
 */
async function cloneVoice() {
  const result = await vocal.clone(
    '/path/to/audio.mp3',  // Audio to clone
    'my_voice_id',         // Custom voice_id
    'Test text',          // Test text
    'speech-2.8-hd'       // Model
  );

  if (result.success) {
    console.log('voice_id:', result.data?.id);
    console.log('preview_url:', result.data?.previewUrl);
  } else {
    console.error('error:', result.error);
  }
}

/**
 * Example 2: Full voice clone
 */
async function cloneVoiceFull() {
  const result = await vocal.cloneVoice({
    audioPath: '/path/to/audio.mp3',
    voiceId: 'my_voice_id',
    promptAudioPath: '/path/to/prompt.mp3',
    promptText: 'Prompt audio text',
    testText: 'Test text',
    model: 'speech-2.8-hd',
  });

  if (result.success) {
    console.log('voice_id:', result.data?.id);
  }
}

/**
 * Example 3: Voice design
 */
async function designVoice() {
  const result = await vocal.design(
    'A deep, resonant narrator with a mysterious atmosphere',
    'Night fell, and the old house stood silent...'
  );

  if (result.success) {
    console.log('voice_id:', result.data?.id);
    console.log('preview_audio:', result.data?.previewAudio);
  }
}

/**
 * Example 4: Speech synthesis
 */
async function speak() {
  const result = await vocal.speak(
    'Hello, this is speech synthesis',
    'my_voice_id'
  );

  if (result.success) {
    console.log('file_id:', result.data?.fileId);
    console.log('url:', result.data?.url);
  }
}

/**
 * Example 5: Full speech synthesis
 */
async function speech() {
  const result = await vocal.speech({
    text: 'Hello, this is speech synthesis',
    voiceId: 'my_voice_id',
    model: 'speech-2.8-hd',
    speed: 1.0,
    volume: 1.0,
    format: 'mp3',
  });

  if (result.success) {
    console.log('file_id:', result.data?.fileId);
    console.log('url:', result.data?.url);
  }
}

// cloneVoice();
// designVoice();
// speak();
