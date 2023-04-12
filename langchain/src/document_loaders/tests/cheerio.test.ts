import { test } from "@jest/globals";
import { CheerioWebBaseLoader } from "../web/cheerio.js";

test("Test cheerio web scraper loader", async () => {
  const loader = new CheerioWebBaseLoader(
    "https://news.ycombinator.com/item?id=34817881"
  );
  await loader.load();
});
