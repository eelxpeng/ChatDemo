import { Message, OpenAIModel } from "@/types";
import { createParser, ParsedEvent, ReconnectInterval } from "eventsource-parser";

export const OpenAIStream = async (model: OpenAIModel, key: string, messages: Message[]) => {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const res = await fetch("https://api.openai.com/v1/chat/completions", {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${key ? key : process.env.OPENAI_API_KEY}`
    },
    method: "POST",
    body: JSON.stringify({
      model,
      messages: [
        {
          role: "system",
          content: `You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown format.`
        },
        ...messages
      ],
      max_tokens: 1000,
      temperature: 0.0,
      stream: true
    })
  });

  if (res.status !== 200) {
    throw new Error("OpenAI API returned an error");
  }

  const stream = new ReadableStream({
    async start(controller) {
      const onParse = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === "event") {
          const data = event.data;

          if (data === "[DONE]") {
            controller.close();
            return;
          }

          try {
            const json = JSON.parse(data);
            const text = json.choices[0].delta.content;
            const queue = encoder.encode(text);
            controller.enqueue(queue);
          } catch (e) {
            controller.error(e);
          }
        }
      };

      const parser = createParser(onParse);

      for await (const chunk of res.body as any) {
        parser.feed(decoder.decode(chunk));
      }
    }
  });

  return stream;
};

export const OpenAIModelURL: Record<OpenAIModel, string> = {
  [OpenAIModel.GPT_3_5]: "http://chat_server_m:80/generate_stream",
  // [OpenAIModel.GPT_4]: "http://chat_server:80/generate_stream"
  // [OpenAIModel.GPT_4]: "GPT-4"
};

// export const DummpyStream = async (model: OpenAIModel, key: string, messages: Message[], temperature: number) => {
//   const encoder = new TextEncoder();
//   const decoder = new TextDecoder();

//   console.log("making request to server")
//   const res = await fetch(OpenAIModelURL[model], {
//     headers: {
//       "Content-Type": "application/json",
//       Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
//     },
//     method: "POST",
//     body: JSON.stringify({
//       model,
//       messages: [
//         {
//           role: "system",
//           content: `Write a response that appropriately completes the request.`
//         },
//         ...messages
//       ],
//       max_tokens: 800,
//       temperature: temperature,
//       stream: true
//     })
//   });

//   if (res.status !== 200) {
//     throw new Error("Dummy Server returned an error");
//   }

//   const stream = res.body

//   return stream;
// };


export const DummpyStream = async (model: OpenAIModel, key: string, messages: Message[], temperature: number) => {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  const format_messages = (messages: Message[]) => {
    let text = "";
    for (let i = 0; i < messages.length; i++) {
      const message = messages[i];
      text += message.role + ":\n" + message.content + "\n";
    }
    // append "assistant:\n"
    text += "assistant:\n";
    return text;
  }

  const text_message = format_messages(messages)
  console.log("making request to server with: \n" + text_message)
  const res = await fetch(OpenAIModelURL[model], {
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
    },
    method: "POST",
    body: JSON.stringify({
      inputs: text_message,
      parameters: {
        temperature: temperature,
        do_sample: temperature > 0 ? true : false,
        max_new_tokens: 800
      },
    })
  });

  if (res.status !== 200) {
    throw new Error("Dummy Server returned an error");
  }

  // const stream = res.body
  const stream = new ReadableStream({
    async start(controller) {
      const onParse = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === "event") {
          const data = event.data;

          try {
            const json = JSON.parse(data);
            if (json.generated_text != null) {
              controller.close();
              return;
            }
            // const text = json.choices[0].delta.content;
            const text = json.token.text;
            const queue = encoder.encode(text);
            controller.enqueue(queue);
          } catch (e) {
            controller.error(e);
          }
        }
      };

      const parser = createParser(onParse);

      for await (const chunk of res.body as any) {
        parser.feed(decoder.decode(chunk));
      }
    }
  });

  return stream;
};
