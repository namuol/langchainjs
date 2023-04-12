import {
  BaseStringPromptTemplate,
  BasePromptTemplateInput,
  BaseExampleSelector,
} from "./base.js";
import {
  TemplateFormat,
  checkValidTemplate,
  renderTemplate,
} from "./template.js";
import { PromptTemplate } from "./prompt.js";
import { SerializedFewShotTemplate } from "./serde.js";
import { Example } from "../schema/index.js";

export interface FewShotPromptTemplateInput<
  K extends string,
  P extends string,
  ExK extends string,
  ExP extends string
> extends BasePromptTemplateInput<K, P> {
  /**
   * Examples to format into the prompt. Exactly one of this or
   * {@link exampleSelector} must be
   * provided.
   */
  examples?: Example<ExK, ExP>[];

  /**
   * An {@link BaseExampleSelector} Examples to format into the prompt. Exactly one of this or
   * {@link examples} must be
   * provided.
   */
  exampleSelector?: BaseExampleSelector<ExK, ExP>;

  /**
   * An {@link PromptTemplate} used to format a single example.
   */
  examplePrompt: PromptTemplate<ExK, ExP>;

  /**
   * String separator used to join the prefix, the examples, and suffix.
   */
  exampleSeparator?: string;

  /**
   * A prompt template string to put before the examples.
   *
   * @defaultValue `""`
   */
  prefix?: string;

  /**
   * A prompt template string to put after the examples.
   */
  suffix?: string;

  /**
   * The format of the prompt template. Options are: 'f-string', 'jinja-2'
   */
  templateFormat?: TemplateFormat;

  /**
   * Whether or not to try validating the template on initialization.
   */
  validateTemplate?: boolean;
}

/**
 * Prompt template that contains few-shot examples.
 * @augments BasePromptTemplate
 * @augments FewShotPromptTemplateInput
 */
export class FewShotPromptTemplate<
    K extends string,
    P extends string,
    ExK extends string,
    ExP extends string
  >
  extends BaseStringPromptTemplate<K, P>
  implements FewShotPromptTemplateInput<K, P, ExK, ExP>
{
  examples?: any[];

  exampleSelector?: BaseExampleSelector<ExK, ExP> | undefined;

  examplePrompt: PromptTemplate<ExK, ExP>;

  suffix = "";

  exampleSeparator = "\n\n";

  prefix = "";

  templateFormat: TemplateFormat = "f-string";

  validateTemplate = true;

  constructor(input: FewShotPromptTemplateInput<K, P, ExK, ExP>) {
    super(input);
    Object.assign(this, input);

    if (this.examples !== undefined && this.exampleSelector !== undefined) {
      throw new Error(
        "Only one of 'examples' and 'example_selector' should be provided"
      );
    }

    if (this.examples === undefined && this.exampleSelector === undefined) {
      throw new Error(
        "One of 'examples' and 'example_selector' should be provided"
      );
    }

    if (this.validateTemplate) {
      let totalInputVariables: string[] = this.inputVariables;
      if (this.partialVariables) {
        totalInputVariables = totalInputVariables.concat(
          Object.keys(this.partialVariables)
        );
      }
      checkValidTemplate(
        this.prefix + this.suffix,
        this.templateFormat,
        totalInputVariables
      );
    }
  }

  _getPromptType(): "few_shot" {
    return "few_shot";
  }

  private async getExamples(inputVariables: any): Promise<any[]> {
    if (this.examples !== undefined) {
      return this.examples;
    }
    if (this.exampleSelector !== undefined) {
      return this.exampleSelector.selectExamples(inputVariables);
    }

    throw new Error(
      "One of 'examples' and 'example_selector' should be provided"
    );
  }

  async partial<P2 extends string>(
    values: Record<P2, any>
  ): Promise<FewShotPromptTemplate<Exclude<K, P2>, P | P2, ExK, ExP>> {
    const promptDict: FewShotPromptTemplate<
      Exclude<K, P2>,
      P | P2,
      ExK,
      ExP
    > = {
      ...this,
    } as never;
    promptDict.inputVariables = this.inputVariables.filter(
      (iv) => !(iv in values)
    ) as Exclude<K, P2>[];
    promptDict.partialVariables = {
      ...(this.partialVariables ?? {}),
      ...values,
    } as Record<P | P2, any>;
    return new FewShotPromptTemplate(promptDict);
  }

  async format(
    values: Record<K, any> & Partial<Record<P, any>>
  ): Promise<string> {
    const allValues = await this.mergePartialAndUserVariables(values);
    const examples = await this.getExamples(allValues);

    const exampleStrings = await Promise.all(
      examples.map((example) => this.examplePrompt.format(example))
    );
    const template = [this.prefix, ...exampleStrings, this.suffix].join(
      this.exampleSeparator
    );
    return renderTemplate(template, this.templateFormat, allValues);
  }

  serialize(): SerializedFewShotTemplate {
    if (this.exampleSelector || !this.examples) {
      throw new Error(
        "Serializing an example selector is not currently supported"
      );
    }
    if (this.outputParser !== undefined) {
      throw new Error(
        "Serializing an output parser is not currently supported"
      );
    }
    return {
      _type: this._getPromptType(),
      input_variables: this.inputVariables,
      example_prompt: this.examplePrompt.serialize(),
      example_separator: this.exampleSeparator,
      suffix: this.suffix,
      prefix: this.prefix,
      template_format: this.templateFormat,
      examples: this.examples,
    };
  }

  static async deserialize(
    data: SerializedFewShotTemplate
  ): Promise<FewShotPromptTemplate<string, string, string, string>> {
    const { example_prompt } = data;
    if (!example_prompt) {
      throw new Error("Missing example prompt");
    }
    const examplePrompt = await PromptTemplate.deserialize(example_prompt);

    let examples: Example<string, string>[];

    if (Array.isArray(data.examples)) {
      examples = data.examples;
    } else {
      throw new Error(
        "Invalid examples format. Only list or string are supported."
      );
    }

    return new FewShotPromptTemplate({
      inputVariables: data.input_variables,
      examplePrompt,
      examples,
      exampleSeparator: data.example_separator,
      prefix: data.prefix,
      suffix: data.suffix,
      templateFormat: data.template_format,
    });
  }
}
