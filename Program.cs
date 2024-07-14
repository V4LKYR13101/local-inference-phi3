//    Copyright (c) 2024
//    Author      : Bruno Capuano
//    Change Log  :
//
//    The MIT License (MIT)
//
//    Permission is hereby granted, free of charge, to any person obtaining a copy
//    of this software and associated documentation files (the "Software"), to deal
//    in the Software without restriction, including without limitation the rights
//    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//    copies of the Software, and to permit persons to whom the Software is
//    furnished to do so, subject to the following conditions:
//
//    The above copyright notice and this permission notice shall be included in
//    all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//    THE SOFTWARE.

using Microsoft.ML.OnnxRuntimeGenAI;
using System.Reflection.Emit;
using System.Reflection;
using System.Speech.Synthesis;
using System.Collections.Concurrent;
using System.Threading;
using static System.Net.Mime.MediaTypeNames;
using System.Speech.Recognition;


internal class Program
{
    private static ConcurrentQueue<string> speakQueue = new ConcurrentQueue<string>();
    private static bool isSpeaking = false;
    private static SemaphoreSlim semaphore = new SemaphoreSlim(0);


    static async Task Main(string[] args)
    {

        string modelPath = @"D:\Models\onnx\cpu-int4-rtn-block-32";
        var model = new Model(modelPath);
        var tokenizer = new Tokenizer(model);
        //_ = Task.Run(ProcessSpeechQueue);
        var systemPrompt = "You are an AI assistant that helps people find information. Answer questions using a direct style. Do not share more information that the requested by the users.";


        // chat start
        Console.WriteLine(@"Ask your question. Type an empty string to Exit.");
        var speak = "";
        // chat loop
        while (true)
        {
            Console.WriteLine();

            Console.Write(@"Q: ");
            
            string userQ = Console.ReadLine();
            if (string.IsNullOrEmpty(userQ))
            {
                break;
            }

            // show phi3 response
            Console.Write("Phi3: ");
            Console.Clear();
            var fullPrompt = $"<|system|>{systemPrompt}<|end|><|user|>{userQ}<|end|><|assistant|>";
            var tokens = tokenizer.Encode(fullPrompt);

            var generatorParams = new GeneratorParams(model);
            generatorParams.SetSearchOption("max_length", 2048);
            generatorParams.SetSearchOption("past_present_share_buffer", false);
            generatorParams.SetInputSequences(tokens);

            var generator = new Generator(model, generatorParams);
            speak = "";
            while (!generator.IsDone())
            {
                generator.ComputeLogits();
                generator.GenerateNextToken();
                var outputTokens = generator.GetSequence(0).ToArray(); // Convert to array immediately
                var newToken = new int[1] { outputTokens[outputTokens.Length - 1] }; // Extract the last token into a new array
                var output = tokenizer.Decode(newToken); // Decode the new tokensArray[outputTokensArray.Length - 1] };
                Console.Write(output);
            }
          

            Console.ReadKey();
        }



    }
}

