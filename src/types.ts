/**
 * Vocal SDK Types
 * 通用的音色和语音合成接口，不与具体 provider 耦合
 */

// ========== 通用接口定义 ==========

/**
 * SDK 配置
 */
export interface VocalConfig {
  apiKey: string;
  baseUrl?: string;
}

/**
 * 音色信息
 */
export interface Voice {
  /**
   * 音色唯一标识
   */
  id: string;
  
  /**
   * 音色名称
   */
  name?: string;
  
  /**
   * 试听的音频 URL
   */
  previewUrl?: string;
  
  /**
   * 试听音频 (可能是 hex 编码)
   */
  previewAudio?: string;
}

/**
 * 音频合成结果
 */
export interface AudioResult {
  /**
   * 文件唯一标识
   */
  fileId: string;
  
  /**
   * 音频文件 URL
   */
  url: string;
}

/**
 * 操作结果
 */
export interface Result<T = void> {
  /**
   * 是否成功
   */
  success: boolean;
  
  /**
   * 数据
   */
  data?: T;
  
  /**
   * 错误信息
   */
  error?: string;
}

/**
 * 音色克隆选项
 */
export interface CloneOptions {
  /**
   * 要克隆的音频文件路径
   */
  audioPath: string;
  
  /**
   * 自定义的 voice_id
   */
  voiceId: string;
  
  /**
   * 参考音频路径 (增强克隆效果)
   */
  promptAudioPath?: string;
  
  /**
   * 参考音频对应的文本
   */
  promptText?: string;
  
  /**
   * 测试文本 (生成试听音频)
   */
  testText?: string;
  
  /**
   * 使用的模型
   */
  model?: string;
}

/**
 * 音色设计选项
 */
export interface DesignOptions {
  /**
   * 音色描述
   */
  prompt: string;
  
  /**
   * 试听音频文本
   */
  previewText: string;
  
  /**
   * 自定义音色ID
   */
  voiceId?: string;
  
  /**
   * 是否添加水印
   */
  watermark?: boolean;
}

/**
 * 语音合成选项
 */
export interface SpeechOptions {
  /**
   * 要合成的文本
   */
  text: string;
  
  /**
   * 音色ID
   */
  voiceId: string;
  
  /**
   * 使用的模型
   */
  model?: string;
  
  /**
   * 语速
   */
  speed?: number;
  
  /**
   * 音量
   */
  volume?: number;
  
  /**
   * 音调
   */
  pitch?: number;
  
  /**
   * 输出格式
   */
  format?: string;
  
  /**
   * 比特率
   */
  bitrate?: number;
  
  /**
   * 采样率
   */
  sampleRate?: number;
}

// ========== MiniMax 特定类型 (内部使用) ==========

/** @internal */
export interface MiniMaxConfig {
  apiKey: string;
  baseUrl?: string;
}

/** @internal */
export interface MiniMaxUploadResponse {
  file: {
    file_id: string;
    purpose: string;
    filename: string;
    size: number;
    created_at: number;
  };
}

/** @internal */
export interface MiniMaxCloneResponse {
  voice_id?: string;
  demo_audio?: string;
  audio?: {
    file_id: string;
    url: string;
  };
  model?: string;
  base_resp?: {
    status_code: number;
    status_msg: string;
  };
}

/** @internal */
export interface MiniMaxT2AResponse {
  data: {
    file_id: string;
    url: string;
  };
}

/** @internal */
export interface MiniMaxDesignResponse {
  voice_id: string;
  trial_audio: string;
  base_resp: {
    status_code: number;
    status_msg: string;
  };
}
