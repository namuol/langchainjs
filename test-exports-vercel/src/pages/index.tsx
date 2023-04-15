// import all entrypoints to test, do not do this in your own app
import "../entrypoints.js";

import Head from "next/head";
import { Inter } from "next/font/google";
import styles from "@/styles/Home.module.css";
import { useCallback } from "react";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { CallbackManager } from "langchain/callbacks";
import { LLMChain } from "langchain/chains";
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
} from "langchain/prompts";

const inter = Inter({ subsets: ["latin"] });

// Don't do this in your app, it would leak your API key
const OPENAI_API_KEY = process.env.NEXT_PUBLIC_OPENAI_API_KEY;

export default function Home() {
  const runChain = useCallback(async () => {
    const llm = new ChatOpenAI({
      openAIApiKey: OPENAI_API_KEY,
      streaming: true,
      callbackManager: CallbackManager.fromHandlers({
        handleLLMNewToken: async (token) =>
          console.log("handleLLMNewToken", token),
      }),
    });

    // Test count tokens
    const n = await llm.getNumTokens("Hello");
    console.log("getNumTokens", n);

    // Test a chain + prompt + model
    const chain = new LLMChain({
      llm,
      prompt: ChatPromptTemplate.fromPromptMessages([
        HumanMessagePromptTemplate.fromTemplate<"input">("{input}"),
      ]),
    });
    const res = await chain.run("hello");

    console.log("runChain", res);
  }, []);

  return (
    <>
      <Head>
        <title>Create Next App</title>
        <meta name="description" content="Generated by create next app" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main className={`${styles.main} ${inter.className}`}>
        <button onClick={runChain}>Click to run a chain</button>
      </main>
    </>
  );
}
