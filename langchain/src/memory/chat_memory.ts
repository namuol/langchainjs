import {
  HumanChatMessage,
  AIChatMessage,
  BaseChatMessage,
  BaseChatMessageHistory,
  InputValues,
} from "../schema/index.js";
import { BaseMemory, OutputValues, getInputValue } from "./base.js";

export class ChatMessageHistory extends BaseChatMessageHistory {
  messages: BaseChatMessage[] = [];

  constructor(messages?: BaseChatMessage[]) {
    super();
    this.messages = messages ?? [];
  }

  addUserMessage(message: string): void {
    this.messages.push(new HumanChatMessage(message));
  }

  addAIChatMessage(message: string): void {
    this.messages.push(new AIChatMessage(message));
  }

  async clear(): Promise<void> {
    this.messages = [];
  }
}

export interface BaseMemoryInput<K extends string, O extends string> {
  chatHistory?: ChatMessageHistory;
  returnMessages?: boolean;
  inputKey?: K;
  outputKey?: O;
}

export abstract class BaseChatMemory<
  K extends string,
  P extends string,
  O extends string,
  MI extends string
> extends BaseMemory<K, P, O, MI> {
  chatHistory: ChatMessageHistory;

  returnMessages = false;

  inputKey?: K;

  outputKey?: O;

  constructor(fields?: BaseMemoryInput<K, O>) {
    super();
    this.chatHistory = fields?.chatHistory ?? new ChatMessageHistory();
    this.returnMessages = fields?.returnMessages ?? this.returnMessages;
    this.inputKey = fields?.inputKey ?? this.inputKey;
    this.outputKey = fields?.outputKey ?? this.outputKey;
  }

  async saveContext(
    inputValues: InputValues<K, P>,
    outputValues: OutputValues<O>
  ): Promise<void> {
    this.chatHistory.addUserMessage(getInputValue(inputValues, this.inputKey));
    this.chatHistory.addAIChatMessage(
      getInputValue(outputValues, this.outputKey)
    );
  }

  async clear(): Promise<void> {
    await this.chatHistory.clear();
  }
}
