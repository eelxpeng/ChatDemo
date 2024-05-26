export enum OpenAIModel {
  // GPT_3_5 = "gpt-3.5-turbo",
  // GPT_3_5_LEGACY = "gpt-3.5-turbo-0301"
  // GPT_4 = "gpt-4"
  GPT_3_5 = "cwgpt-large-chat",
  // GPT_4 = "cwgpt-xlarge-chat",
}

export const OpenAIModelNames: Record<OpenAIModel, string> = {
  [OpenAIModel.GPT_3_5]: "Default Custom Chat LLM",
  // [OpenAIModel.GPT_4]: "CWGPT-XLarge-Chat"
  // [OpenAIModel.GPT_4]: "GPT-4"
};

export interface Message {
  role: Role;
  content: string;
}

export type Role = "assistant" | "user";

export interface Conversation {
  id: number;
  name: string;
  messages: Message[];
}
