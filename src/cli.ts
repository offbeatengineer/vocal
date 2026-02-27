#!/usr/bin/env node

/**
 * Vocal CLI - 音色克隆和语音合成工具
 */

import { Command } from 'commander';
import { createVocal, Vocal } from './index';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';

/**
 * 下载文件
 */
async function downloadFile(url: string, outputPath: string): Promise<void> {
  const response = await axios.get(url, { responseType: 'arraybuffer' });
  fs.writeFileSync(outputPath, Buffer.from(response.data));
  console.log(`   试听音频已保存: ${outputPath}`);
}

/**
 * 解析 hex 音频并保存为文件
 */
function saveHexAudio(hexAudio: string, outputPath: string): void {
  const hex = hexAudio.replace(/\s/g, '');
  const buffer = Buffer.from(hex, 'hex');
  fs.writeFileSync(outputPath, buffer);
  console.log(`   试听音频已保存: ${outputPath}`);
}

/**
 * 加载配置
 */
function loadConfig() {
  const envFile = path.join(process.cwd(), '.env');
  const envVars: Record<string, string> = {};
  
  if (fs.existsSync(envFile)) {
    const content = fs.readFileSync(envFile, 'utf-8');
    content.split('\n').forEach((line) => {
      line = line.trim();
      if (!line || line.startsWith('#')) return;
      
      const eqIndex = line.indexOf('=');
      if (eqIndex === -1) return;

      const key = line.slice(0, eqIndex).trim();
      let value = line.slice(eqIndex + 1).trim();

      if ((value.startsWith('"') && value.endsWith('"')) ||
          (value.startsWith("'") && value.endsWith("'"))) {
        value = value.slice(1, -1);
      }

      envVars[key] = value;
    });
  }
  
  const apiKey = envVars.MINIMAX_API_KEY || process.env.MINIMAX_API_KEY;
  
  if (!apiKey) {
    console.error('❌ 请设置 API Key');
    console.error('');
    console.error('方式1: 环境变量');
    console.error('   export MINIMAX_API_KEY="your-api-key"');
    console.error('');
    console.error('方式2: .env 文件 (当前目录)');
    console.error('   echo "MINIMAX_API_KEY=your-api-key" > .env');
    process.exit(1);
  }

  const baseUrl = envVars.MINIMAX_BASE_URL || process.env.MINIMAX_BASE_URL;
  
  return { apiKey, baseUrl };
}

// 主程序
const program = new Command();

program
  .name('vocal')
  .description('音色克隆和语音合成工具')
  .version('1.0.0');

// clone 命令
program
  .command('clone')
  .description('克隆音色')
  .requiredOption('-a, --audio <path>', '待克隆的音频文件路径 (mp3/m4a/wav)')
  .requiredOption('-v, --voice-id <id>', '自定义的 voice_id')
  .option('-p, --prompt-audio <path>', '参考音频路径 (增强克隆效果)')
  .option('-t, --prompt-text <text>', '参考音频对应的文本')
  .option('--text <text>', '测试文本')
  .option('-m, --model <model>', '使用的模型 (speech-2.8 或 speech-2.8-hd)', 'speech-2.8-hd')
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    const result = await vocal.cloneVoice({
      audioPath: options.audio,
      voiceId: options.voiceId,
      promptAudioPath: options.promptAudio,
      promptText: options.promptText,
      testText: options.text,
      model: options.model,
    });

    if (!result.success) {
      console.error('❌ 克隆失败:', result.error);
      process.exit(1);
    }

    console.log('🎉 音色克隆成功!');
    console.log('');
    console.log('📋 结果:');
    console.log(`   voice_id: ${result.data?.id}`);
    
    // 下载预览音频
    if (result.data?.previewUrl) {
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_clone_preview.mp3`);
      try {
        await downloadFile(result.data.previewUrl, outputPath);
      } catch (error: any) {
        console.log(`   下载失败: ${error.message}`);
        console.log(`   试听音频: ${result.data.previewUrl}`);
      }
    } else if (result.data?.previewAudio) {
      // design 命令返回的是 hex 编码
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_preview.mp3`);
      saveHexAudio(result.data.previewAudio, outputPath);
    } else {
      console.log('   (无试听音频)');
    }
    
    console.log('');
    console.log('💡 使用示例:');
    console.log(`   vocal speak -t "要合成的文本" -v ${result.data?.id}`);
    console.log('');
  });

// tts 命令 (speak 的别名)
program
  .command('tts')
  .description('语音合成')
  .requiredOption('-t, --text <text>', '要合成的文本')
  .requiredOption('-v, --voice-id <id>', 'voice_id')
  .option('-m, --model <model>', '使用的模型', 'speech-2.8-hd')
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    const result = await vocal.speech({
      text: options.text,
      voiceId: options.voiceId,
      model: options.model,
    });

    if (!result.success) {
      console.error('❌ 合成失败:', result.error);
      process.exit(1);
    }

    console.log('🎉 语音合成成功!');
    console.log('');
    console.log('📋 结果:');
    console.log(`   文件ID: ${result.data?.fileId}`);
    console.log(`   音频链接: ${result.data?.url}`);
    console.log('');
  });

// speak 命令
program
  .command('speak')
  .description('语音合成')
  .requiredOption('-t, --text <text>', '要合成的文本')
  .requiredOption('-v, --voice-id <id>', 'voice_id')
  .option('-m, --model <model>', '使用的模型', 'speech-2.8-hd')
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    const result = await vocal.speech({
      text: options.text,
      voiceId: options.voiceId,
      model: options.model,
    });

    if (!result.success) {
      console.error('❌ 合成失败:', result.error);
      process.exit(1);
    }

    console.log('🎉 语音合成成功!');
    console.log('');
    console.log('📋 结果:');
    console.log(`   文件ID: ${result.data?.fileId}`);
    console.log(`   音频链接: ${result.data?.url}`);
    console.log('');
  });

// upload 命令
program
  .command('upload')
  .description('上传音频文件')
  .requiredOption('-a, --audio <path>', '音频文件路径')
  .option('--purpose <purpose>', '用途 (voice_clone/prompt_audio)', 'voice_clone')
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    try {
      let fileId: string;
      if (options.purpose === 'prompt_audio') {
        fileId = await vocal.uploadPromptAudio(options.audio);
      } else {
        fileId = await vocal.uploadCloneAudio(options.audio);
      }

      console.log('✅ 上传成功!');
      console.log(`   file_id: ${fileId}`);
      console.log('');
    } catch (error: any) {
      console.error('❌ 上传失败:', error.message);
      process.exit(1);
    }
  });

// design 命令
program
  .command('design')
  .description('音色设计 (通过文字描述生成音色)')
  .requiredOption('-p, --prompt <text>', '音色描述，例如：讲述悬疑故事的播音员，声音低沉富有磁性')
  .requiredOption('-t, --text <text>', '试听音频文本 (最多500字符)')
  .option('-v, --voice-id <id>', '自定义音色ID (不传则自动生成)')
  .option('--watermark', '在试听音频末尾添加水印', false)
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    console.log('🎨 正在设计音色...');
    
    const result = await vocal.designVoice({
      prompt: options.prompt,
      previewText: options.text,
      voiceId: options.voiceId,
      watermark: options.watermark,
    });

    if (!result.success) {
      console.error('❌ 音色设计失败:', result.error);
      process.exit(1);
    }

    console.log('');
    console.log('🎉 音色设计成功!');
    console.log('');
    console.log('📋 结果:');
    console.log(`   voice_id: ${result.data?.id}`);
    
    if (result.data?.previewAudio) {
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_preview.mp3`);
      saveHexAudio(result.data.previewAudio, outputPath);
    }
    
    console.log('');
    console.log('💡 使用示例:');
    console.log(`   vocal speak -t "要合成的文本" -v ${result.data?.id}`);
    console.log('');
  });

program.parse();
