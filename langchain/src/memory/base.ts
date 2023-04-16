import { BaseChatMessage, ChatMessage, InputValues } from "../schema/index.js";
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type OutputValues<K extends string = string> = Record<K, any>;
export type MemoryVariables<
  K extends string = string,
  P extends string = string
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
> = Record<K, any> & Partial<Record<P, any>>;

export abstract class BaseMemory<
  K extends string,
  P extends string,
  O extends string,
  MI extends string
> {
  abstract loadMemoryVariables(
    values: InputValues<K, P>
  ): Promise<MemoryVariables<MI>>;

  abstract saveContext(
    inputValues: InputValues<K, P>,
    outputValues: OutputValues<O>
  ): Promise<void>;
}

export const getInputValue = <K extends string, P extends string>(
  inputValues: InputValues<K, P>,
  inputKey?: K
) => {
  if (inputKey !== undefined) {
    return inputValues[inputKey];
  }
  const keys = Object.keys(inputValues) as (K | P)[];
  if (keys.length === 1) {
    return inputValues[keys[0]];
  }
  throw new Error(
    `input values have multiple keys, memory only supported when one key currently: ${keys}`
  );
};

export function getBufferString(
  messages: BaseChatMessage[],
  human_prefix = "Human",
  ai_prefix = "AI"
): string {
  const string_messages: string[] = [];
  for (const m of messages) {
    let role: string;
    if (m._getType() === "human") {
      role = human_prefix;
    } else if (m._getType() === "ai") {
      role = ai_prefix;
    } else if (m._getType() === "system") {
      role = "System";
    } else if (m._getType() === "generic") {
      role = (m as ChatMessage).role;
    } else {
      throw new Error(`Got unsupported message type: ${m}`);
    }
    string_messages.push(`${role}: ${m.text}`);
  }
  return string_messages.join("\n");
}
