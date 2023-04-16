import { BaseChatModel } from "../chat_models/base.js";
import { BasePromptTemplate } from "../prompts/base.js";
import { BaseLanguageModel } from "../base_language/index.js";
import { BaseLLM } from "../llms/base.js";

export abstract class BasePromptSelector<K extends string, P extends string> {
  abstract getPrompt(llm: BaseLanguageModel): BasePromptTemplate<K, P>;
}

export class ConditionalPromptSelector<
  K extends string,
  P extends string
> extends BasePromptSelector<K, P> {
  defaultPrompt: BasePromptTemplate<K, P>;

  conditionals: Array<
    [
      condition: (llm: BaseLanguageModel) => boolean,
      prompt: BasePromptTemplate<K, P>
    ]
  >;

  constructor(
    default_prompt: BasePromptTemplate<K, P>,
    conditionals: Array<
      [
        condition: (llm: BaseLanguageModel) => boolean,
        prompt: BasePromptTemplate<K, P>
      ]
    > = []
  ) {
    super();
    this.defaultPrompt = default_prompt;
    this.conditionals = conditionals;
  }

  getPrompt(llm: BaseLanguageModel): BasePromptTemplate<K, P> {
    for (const [condition, prompt] of this.conditionals) {
      if (condition(llm)) {
        return prompt;
      }
    }
    return this.defaultPrompt;
  }
}

export function isLLM(llm: BaseLanguageModel): llm is BaseLLM {
  return llm._modelType() === "base_llm";
}

export function isChatModel(llm: BaseLanguageModel): llm is BaseChatModel {
  return llm._modelType() === "base_chat_model";
}
