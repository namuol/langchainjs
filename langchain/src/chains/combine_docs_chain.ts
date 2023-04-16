import type {
  SerializedStuffDocumentsChain,
  SerializedMapReduceDocumentsChain,
  SerializedRefineDocumentsChain,
} from "./serde.js";
import { BaseChain } from "./base.js";
import { LLMChain } from "./llm_chain.js";

import { Document } from "../document.js";

import { ChainValues } from "../schema/index.js";
import { BasePromptTemplate } from "../prompts/base.js";
import { PromptTemplate } from "../prompts/prompt.js";

export interface StuffDocumentsChainInput<
  LK extends string,
  LP extends string,
  LO extends string,
  LMI extends string,
  K extends string
> {
  /** LLM Wrapper to use after formatting documents */
  llmChain: LLMChain<LK, LP, LO, LMI>;
  inputKey: K;
  outputKey: string;
  /** Variable name in the LLM chain to put the documents in */
  documentVariableName: LK | LP;
}

/**
 * Chain that combines documents by stuffing into context.
 * @augments BaseChain
 * @augments StuffDocumentsChainInput
 */
export class StuffDocumentsChain<
    LK extends string,
    LP extends string,
    LO extends string,
    LMI extends string,
    K extends string = "input_documents"
  >
  extends BaseChain<K, string, LO, string>
  implements StuffDocumentsChainInput<LK, LP, LO, LMI, K>
{
  llmChain: LLMChain<LK, LP, LO, LMI>;

  inputKey = "input_documents" as K;

  outputKey = "output_text";

  documentVariableName = "context" as LK | LP;

  get inputKeys() {
    return [this.inputKey, ...this.llmChain.inputKeys];
  }

  constructor(fields: {
    llmChain: LLMChain<LK, LP, LO, LMI>;
    inputKey?: K;
    outputKey?: string;
    documentVariableName?: LK | LP;
  }) {
    super();
    this.llmChain = fields.llmChain;
    this.documentVariableName =
      fields.documentVariableName ?? this.documentVariableName;
    this.inputKey = fields.inputKey ?? this.inputKey;
    this.outputKey = fields.outputKey ?? this.outputKey;
  }

  async _call(values: ChainValues<K, string>): Promise<ChainValues<LO, never>> {
    if (!(this.inputKey in values)) {
      throw new Error(`Document key ${this.inputKey} not found.`);
    }
    const { [this.inputKey]: docs, ...rest } = values;
    const texts = (docs as Document[]).map(({ pageContent }) => pageContent);
    const text = texts.join("\n\n");
    const result = await this.llmChain.call({
      ...rest,
      [this.documentVariableName]: text,
    } as ChainValues<LK, LP>);
    return result;
  }

  _chainType() {
    return "stuff_documents_chain" as const;
  }

  static async deserialize(data: SerializedStuffDocumentsChain) {
    if (!data.llm_chain) {
      throw new Error("Missing llm_chain");
    }

    return new StuffDocumentsChain({
      llmChain: await LLMChain.deserialize(data.llm_chain),
    });
  }

  serialize(): SerializedStuffDocumentsChain {
    return {
      _type: this._chainType(),
      llm_chain: this.llmChain.serialize(),
    };
  }
}

export interface MapReduceDocumentsChainInput
  extends StuffDocumentsChainInput<any, any, any, any, any> {
  maxTokens: number;
  maxIterations: number;
  combineDocumentsChain: BaseChain<any, any, any, any>;
}

/**
 * Combine documents by mapping a chain over them, then combining results.
 * @augments BaseChain
 * @augments StuffDocumentsChainInput
 */
export class MapReduceDocumentsChain
  extends BaseChain<any, any, any, any>
  implements StuffDocumentsChainInput<any, any, any, any, any>
{
  llmChain: LLMChain<any, any, any, any>;

  inputKey = "input_documents";

  outputKey = "output_text";

  documentVariableName = "context";

  get inputKeys() {
    return [this.inputKey, ...this.combineDocumentChain.inputKeys];
  }

  maxTokens = 3000;

  maxIterations = 10;

  ensureMapStep = false;

  combineDocumentChain: BaseChain<any, any, any, any>;

  constructor(fields: {
    llmChain: LLMChain<any, any, any, any>;
    combineDocumentChain: BaseChain<any, any, any, any>;
    ensureMapStep?: boolean;
    inputKey?: string;
    outputKey?: string;
    documentVariableName?: string;
    maxTokens?: number;
    maxIterations?: number;
  }) {
    super();
    this.llmChain = fields.llmChain;
    this.combineDocumentChain = fields.combineDocumentChain;
    this.documentVariableName =
      fields.documentVariableName ?? this.documentVariableName;
    this.ensureMapStep = fields.ensureMapStep ?? this.ensureMapStep;
    this.inputKey = fields.inputKey ?? this.inputKey;
    this.outputKey = fields.outputKey ?? this.outputKey;
    this.maxTokens = fields.maxTokens ?? this.maxTokens;
    this.maxIterations = fields.maxIterations ?? this.maxIterations;
  }

  async _call(values: ChainValues<any, any>): Promise<ChainValues<any, any>> {
    if (!(this.inputKey in values)) {
      throw new Error(`Document key ${this.inputKey} not found.`);
    }
    const { [this.inputKey]: docs, ...rest } = values;

    let currentDocs = docs as Document[];

    for (let i = 0; i < this.maxIterations; i += 1) {
      const inputs = currentDocs.map((d) => ({
        [this.documentVariableName]: d.pageContent,
        ...rest,
      }));
      const promises = inputs.map(async (i) => {
        const prompt = await this.llmChain.prompt.format(i);
        return this.llmChain.llm.getNumTokens(prompt);
      });

      const length = await Promise.all(promises).then((results) =>
        results.reduce((a, b) => a + b, 0)
      );

      const canSkipMapStep = i !== 0 || !this.ensureMapStep;
      const withinTokenLimit = length < this.maxTokens;
      if (canSkipMapStep && withinTokenLimit) {
        break;
      }

      const results = await this.llmChain.apply(inputs);
      const { outputKey } = this.llmChain;

      // FIXME: Missing `metadata` here; should this be optional on `Document`?
      currentDocs = results.map((r: ChainValues<any, any>) => ({
        pageContent: r[outputKey],
      })) as Document[];
    }
    const newInputs = { input_documents: currentDocs, ...rest };
    const result = await this.combineDocumentChain.call(newInputs);
    return result;
  }

  _chainType() {
    return "map_reduce_documents_chain" as const;
  }

  static async deserialize(data: SerializedMapReduceDocumentsChain) {
    if (!data.llm_chain) {
      throw new Error("Missing llm_chain");
    }

    if (!data.combine_document_chain) {
      throw new Error("Missing combine_document_chain");
    }

    return new MapReduceDocumentsChain({
      llmChain: await LLMChain.deserialize(data.llm_chain),
      combineDocumentChain: await BaseChain.deserialize(
        data.combine_document_chain
      ),
    });
  }

  serialize(): SerializedMapReduceDocumentsChain {
    return {
      _type: this._chainType(),
      llm_chain: this.llmChain.serialize(),
      combine_document_chain: this.combineDocumentChain.serialize(),
    };
  }
}

export interface RefineDocumentsChainInput
  extends StuffDocumentsChainInput<any, any, any, any, any> {
  refineLLMChain: LLMChain<any, any, any, any>;
  documentPrompt: BasePromptTemplate<any, any>;
}

/**
 * Combine documents by doing a first pass and then refining on more documents.
 * @augments BaseChain
 * @augments RefineDocumentsChainInput
 */
export class RefineDocumentsChain
  extends BaseChain<any, any, any, any>
  implements RefineDocumentsChainInput
{
  llmChain: LLMChain<any, any, any, any>;

  inputKey = "input_documents";

  outputKey = "output_text";

  documentVariableName = "context";

  initialResponseName = "existing_answer";

  refineLLMChain: LLMChain<any, any, any, any>;

  get defaultDocumentPrompt(): BasePromptTemplate<any, any> {
    return new PromptTemplate({
      inputVariables: ["page_content"],
      template: "{page_content}",
    });
  }

  documentPrompt = this.defaultDocumentPrompt;

  get inputKeys() {
    return [this.inputKey, ...this.refineLLMChain.inputKeys];
  }

  constructor(fields: {
    llmChain: LLMChain<any, any, any, any>;
    refineLLMChain: LLMChain<any, any, any, any>;
    inputKey?: string;
    outputKey?: string;
    documentVariableName?: string;
    documentPrompt?: BasePromptTemplate<any, any>;
    initialResponseName?: string;
  }) {
    super();
    this.llmChain = fields.llmChain;
    this.refineLLMChain = fields.refineLLMChain;
    this.documentVariableName =
      fields.documentVariableName ?? this.documentVariableName;
    this.inputKey = fields.inputKey ?? this.inputKey;
    this.documentPrompt = fields.documentPrompt ?? this.documentPrompt;
    this.initialResponseName =
      fields.initialResponseName ?? this.initialResponseName;
  }

  async _constructInitialInputs(doc: Document, rest: Record<string, unknown>) {
    const baseInfo: Record<string, unknown> = {
      page_content: doc.pageContent,
      ...doc.metadata,
    };
    const documentInfo: Record<string, unknown> = {};
    this.documentPrompt.inputVariables.forEach((value) => {
      documentInfo[value] = baseInfo[value];
    });

    const baseInputs: Record<string, unknown> = {
      [this.documentVariableName]: await this.documentPrompt.format({
        ...documentInfo,
      }),
    };
    const inputs = { ...baseInputs, ...rest };
    return inputs;
  }

  async _constructRefineInputs(doc: Document, res: string) {
    const baseInfo: Record<string, unknown> = {
      page_content: doc.pageContent,
      ...doc.metadata,
    };
    const documentInfo: Record<string, unknown> = {};
    this.documentPrompt.inputVariables.forEach((value) => {
      documentInfo[value] = baseInfo[value];
    });
    const baseInputs: Record<string, unknown> = {
      [this.documentVariableName]: await this.documentPrompt.format({
        ...documentInfo,
      }),
    };
    const inputs = { [this.initialResponseName]: res, ...baseInputs };
    return inputs;
  }

  async _call(values: ChainValues<any, any>): Promise<ChainValues<any, any>> {
    if (!(this.inputKey in values)) {
      throw new Error(`Document key ${this.inputKey} not found.`);
    }
    const { [this.inputKey]: docs, ...rest } = values;

    const currentDocs = docs as Document[];

    const initialInputs = await this._constructInitialInputs(
      currentDocs[0],
      rest
    );
    let res = await this.llmChain.predict({ ...initialInputs });

    const refineSteps = [res];

    for (let i = 1; i < currentDocs.length; i += 1) {
      const refineInputs = await this._constructRefineInputs(
        currentDocs[i],
        res
      );
      const inputs = { ...refineInputs, ...rest };
      res = await this.refineLLMChain.predict({ ...inputs });
      refineSteps.push(res);
    }

    return { [this.outputKey]: res };
  }

  _chainType() {
    return "refine_documents_chain" as const;
  }

  static async deserialize(data: SerializedRefineDocumentsChain) {
    const SerializedLLMChain = data.llm_chain;

    if (!SerializedLLMChain) {
      throw new Error("Missing llm_chain");
    }

    const SerializedRefineDocumentChain = data.refine_llm_chain;

    if (!SerializedRefineDocumentChain) {
      throw new Error("Missing refine_llm_chain");
    }

    return new RefineDocumentsChain({
      llmChain: await LLMChain.deserialize(SerializedLLMChain),
      refineLLMChain: await LLMChain.deserialize(SerializedRefineDocumentChain),
    });
  }

  serialize(): SerializedRefineDocumentsChain {
    return {
      _type: this._chainType(),
      llm_chain: this.llmChain.serialize(),
      refine_llm_chain: this.refineLLMChain.serialize(),
    };
  }
}
