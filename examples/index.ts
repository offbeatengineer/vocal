/**
 * Vocal SDK 使用示例
 */

import { createVocal, Vocal } from 'vocal';

// 创建实例
const vocal = createVocal(process.env.MINIMAX_API_KEY || 'your-api-key');

/**
 * 示例1: 简化版音色克隆
 */
async function cloneVoice() {
  const result = await vocal.clone(
    '/path/to/audio.mp3',  // 要克隆的音频
    'my_voice_id',          // 自定义 voice_id
    '测试文本',             // 测试文本
    'speech-2.8-hd'        // 模型
  );

  if (result.success) {
    console.log('voice_id:', result.data?.id);
    console.log('preview_url:', result.data?.previewUrl);
  } else {
    console.error('error:', result.error);
  }
}

/**
 * 示例2: 完整版音色克隆
 */
async function cloneVoiceFull() {
  const result = await vocal.cloneVoice({
    audioPath: '/path/to/audio.mp3',
    voiceId: 'my_voice_id',
    promptAudioPath: '/path/to/prompt.mp3',
    promptText: '参考音频文本',
    testText: '测试文本',
    model: 'speech-2.8-hd',
  });

  if (result.success) {
    console.log('voice_id:', result.data?.id);
  }
}

/**
 * 示例3: 音色设计
 */
async function designVoice() {
  const result = await vocal.design(
    '讲述悬疑故事的播音员，声音低沉富有磁性',
    '夜深了，古屋里只有他一人...'
  );

  if (result.success) {
    console.log('voice_id:', result.data?.id);
    console.log('preview_audio:', result.data?.previewAudio);
  }
}

/**
 * 示例4: 语音合成
 */
async function speak() {
  const result = await vocal.speak(
    '你好，这是语音合成',
    'my_voice_id'
  );

  if (result.success) {
    console.log('file_id:', result.data?.fileId);
    console.log('url:', result.data?.url);
  }
}

/**
 * 示例5: 完整版语音合成
 */
async function speech() {
  const result = await vocal.speech({
    text: '你好，这是语音合成',
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
