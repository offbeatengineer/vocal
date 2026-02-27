#!/usr/bin/env node

/**
 * Vocal CLI - Voice cloning and speech synthesis tool
 */

import { Command } from 'commander';
import { createVocal, Vocal } from './index';
import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';

/**
 * Download file from URL
 */
async function downloadFile(url: string, outputPath: string): Promise<void> {
  const response = await axios.get(url, { responseType: 'arraybuffer' });
  fs.writeFileSync(outputPath, Buffer.from(response.data));
  console.log(`   Preview audio saved: ${outputPath}`);
}

/**
 * Save hex audio to file
 */
function saveHexAudio(hexAudio: string, outputPath: string): void {
  const hex = hexAudio.replace(/\s/g, '');
  const buffer = Buffer.from(hex, 'hex');
  fs.writeFileSync(outputPath, buffer);
  console.log(`   Preview audio saved: ${outputPath}`);
}

/**
 * Load config from .env file
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
    console.error('Please set API Key');
    console.error('');
    console.error('Option 1: Environment variable');
    console.error('   export MINIMAX_API_KEY="your-api-key"');
    console.error('');
    console.error('Option 2: .env file (current directory)');
    console.error('   echo "MINIMAX_API_KEY=your-api-key" > .env');
    process.exit(1);
  }

  const baseUrl = envVars.MINIMAX_BASE_URL || process.env.MINIMAX_BASE_URL;
  
  return { apiKey, baseUrl };
}

// Main program
const program = new Command();

program
  .name('vocal')
  .description('Voice cloning and speech synthesis tool')
  .version('1.0.0');

// clone command
program
  .command('clone')
  .description('Clone voice')
  .requiredOption('-a, --audio <path>', 'Audio file path to clone (mp3/m4a/wav)')
  .requiredOption('-v, --voice-id <id>', 'Custom voice_id')
  .option('-p, --prompt-audio <path>', 'Prompt audio path (enhance cloning effect)')
  .option('-t, --prompt-text <text>', 'Prompt audio text')
  .option('--text <text>', 'Test text')
  .option('-m, --model <model>', 'Model (speech-2.8 or speech-2.8-hd)', 'speech-2.8-hd')
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
      console.error('Clone failed:', result.error);
      process.exit(1);
    }

    console.log('Voice cloned successfully!');
    console.log('');
    console.log('Result:');
    console.log(`   voice_id: ${result.data?.id}`);
    
    // Download preview audio
    if (result.data?.previewUrl) {
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_clone_preview.mp3`);
      try {
        await downloadFile(result.data.previewUrl, outputPath);
      } catch (error: any) {
        console.log(`   Download failed: ${error.message}`);
        console.log(`   Preview audio: ${result.data.previewUrl}`);
      }
    } else if (result.data?.previewAudio) {
      // design command returns hex encoded audio
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_preview.mp3`);
      saveHexAudio(result.data.previewAudio, outputPath);
    } else {
      console.log('   (No preview audio)');
    }
    
    console.log('');
    console.log('Usage example:');
    console.log(`   vocal speak -t "Text to speak" -v ${result.data?.id}`);
    console.log('');
  });

// tts command (alias for speak)
program
  .command('tts')
  .description('Speech synthesis')
  .requiredOption('-t, --text <text>', 'Text to synthesize')
  .requiredOption('-v, --voice-id <id>', 'voice_id')
  .option('-m, --model <model>', 'Model', 'speech-2.8-hd')
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
      console.error('Synthesis failed:', result.error);
      process.exit(1);
    }

    console.log('Speech synthesized successfully!');
    console.log('');
    console.log('Result:');
    console.log(`   file_id: ${result.data?.fileId}`);
    console.log(`   audio_url: ${result.data?.url}`);
    console.log('');
  });

// speak command
program
  .command('speak')
  .description('Speech synthesis')
  .requiredOption('-t, --text <text>', 'Text to synthesize')
  .requiredOption('-v, --voice-id <id>', 'voice_id')
  .option('-m, --model <model>', 'Model', 'speech-2.8-hd')
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
      console.error('Synthesis failed:', result.error);
      process.exit(1);
    }

    console.log('Speech synthesized successfully!');
    console.log('');
    console.log('Result:');
    console.log(`   file_id: ${result.data?.fileId}`);
    console.log(`   audio_url: ${result.data?.url}`);
    console.log('');
  });

// upload command
program
  .command('upload')
  .description('Upload audio file')
  .requiredOption('-a, --audio <path>', 'Audio file path')
  .option('--purpose <purpose>', 'Purpose (voice_clone/prompt_audio)', 'voice_clone')
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

      console.log('Upload successful!');
      console.log(`   file_id: ${fileId}`);
      console.log('');
    } catch (error: any) {
      console.error('Upload failed:', error.message);
      process.exit(1);
    }
  });

// design command
program
  .command('design')
  .description('Design voice (generate voice from text description)')
  .requiredOption('-p, --prompt <text>', 'Voice description, e.g.: A deep, resonant narrator with a mysterious atmosphere')
  .requiredOption('-t, --text <text>', 'Preview text (max 500 characters)')
  .option('-v, --voice-id <id>', 'Custom voice_id (auto-generated if not provided)')
  .option('--watermark', 'Add watermark to preview audio', false)
  .action(async (options) => {
    const { apiKey, baseUrl } = loadConfig();
    const vocal = createVocal(apiKey, baseUrl);
    
    console.log('');

    console.log('Designing voice...');
    
    const result = await vocal.designVoice({
      prompt: options.prompt,
      previewText: options.text,
      voiceId: options.voiceId,
      watermark: options.watermark,
    });

    if (!result.success) {
      console.error('Voice design failed:', result.error);
      process.exit(1);
    }

    console.log('');
    console.log('Voice designed successfully!');
    console.log('');
    console.log('Result:');
    console.log(`   voice_id: ${result.data?.id}`);
    
    if (result.data?.previewAudio) {
      const voiceId = result.data.id;
      const outputPath = path.join(process.cwd(), `${voiceId}_preview.mp3`);
      saveHexAudio(result.data.previewAudio, outputPath);
    }
    
    console.log('');
    console.log('Usage example:');
    console.log(`   vocal speak -t "Text to speak" -v ${result.data?.id}`);
    console.log('');
  });

program.parse();
