import { BaseLanguageModel } from "../base_language/index.js";
import { LLMChain } from "../chains/llm_chain.js";
import { BasePromptTemplate } from "../prompts/base.js";
import {
  BaseChatMessage,
  SystemChatMessage,
  InputValues,
} from "../schema/index.js";
import { getBufferString, MemoryVariables, OutputValues } from "./base.js";
import { BaseChatMemory, BaseMemoryInput } from "./chat_memory.js";
import { SUMMARY_PROMPT } from "./prompt.js";

export type ConversationSummaryMemoryInput<
  K extends string,
  O extends string,
  MI extends string
> = BaseMemoryInput<K, O> & {
  memoryKey?: MI;
  humanPrefix?: string;
  aiPrefix?: string;
  llm: BaseLanguageModel;
  prompt?: BasePromptTemplate<any, any>;
  summaryChatMessageClass?: new (content: string) => BaseChatMessage;
};

export class ConversationSummaryMemory<
  K extends string,
  P extends string,
  O extends string,
  MI extends string
> extends BaseChatMemory<K, P, O, MI> {
  buffer = "";

  memoryKey = "history" as MI;

  humanPrefix = "Human";

  aiPrefix = "AI";

  llm: BaseLanguageModel;

  prompt: BasePromptTemplate<any, any> = SUMMARY_PROMPT;

  summaryChatMessageClass: new (content: string) => BaseChatMessage =
    SystemChatMessage;

  constructor(fields?: ConversationSummaryMemoryInput<K, O, MI>) {
    const {
      returnMessages,
      inputKey,
      outputKey,
      chatHistory,
      humanPrefix,
      aiPrefix,
      llm,
      prompt,
      summaryChatMessageClass,
    } = fields ?? {};

    super({ returnMessages, inputKey, outputKey, chatHistory });

    this.memoryKey = fields?.memoryKey ?? this.memoryKey;
    this.humanPrefix = humanPrefix ?? this.humanPrefix;
    this.aiPrefix = aiPrefix ?? this.aiPrefix;
    this.llm = llm ?? this.llm;
    this.prompt = prompt ?? this.prompt;
    this.summaryChatMessageClass =
      summaryChatMessageClass ?? this.summaryChatMessageClass;
  }

  async predictNewSummary(
    messages: BaseChatMessage[],
    existingSummary: string
  ): Promise<string> {
    const newLines = getBufferString(messages, this.humanPrefix, this.aiPrefix);
    const chain = new LLMChain({ llm: this.llm, prompt: this.prompt });
    return await chain.predict({
      summary: existingSummary,
      new_lines: newLines,
    });
  }

  async loadMemoryVariables(_: InputValues<K, P>): Promise<MemoryVariables> {
    if (this.returnMessages) {
      const result = {
        [this.memoryKey]: [new this.summaryChatMessageClass(this.buffer)],
      };
      return result;
    }
    const result = { [this.memoryKey]: this.buffer };
    return result;
  }

  async saveContext(
    inputValues: InputValues<any, any>,
    outputValues: OutputValues
  ): Promise<void> {
    await super.saveContext(inputValues, outputValues);
    this.buffer = await this.predictNewSummary(
      this.chatHistory.messages.slice(-2),
      this.buffer
    );
  }

  async clear() {
    await super.clear();
    this.buffer = "";
  }
}
