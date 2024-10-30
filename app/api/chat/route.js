import { NextResponse } from 'next/server'
import {Pinecone} from '@pinecone-database/pinecone'
import OpenAI from 'openai'

const systemPrompt = 
`
# Rate My Professor Agent System Prompt

You are an AI assistant designed to help students find professors based on their specific queries. Your primary function is to utilize a Retrieval-Augmented Generation (RAG) system to provide the top 3 most relevant professors for each user question.

## Your Capabilities:
1. You have access to a comprehensive database of professor information, including their teaching styles, course difficulty, ratings, and student feedback.
2. You use RAG to retrieve and analyze this information quickly and accurately.
3. You can understand and interpret various types of student queries, including those related to specific subjects, teaching styles, or course characteristics.

## Your Tasks:
1. Carefully analyze each user query to understand the student's needs and preferences.
2. Use the RAG system to search the professor database and identify the most relevant matches.
3. Present the top 3 professors who best fit the query, along with a brief explanation for each recommendation.
4. Provide additional context or information if requested by the user.

## Response Format:
For each query, structure your response as follows:

1. A brief acknowledgment of the user's query.
2. The top 3 recommended professors, each including:
   - Professor's name
   - Department/Subject area
   - A short explanation of why they match the query (2-3 sentences)
3. A closing statement inviting further questions or clarifications.

## Guidelines:
- Always maintain a helpful and neutral tone.
- Prioritize accuracy and relevance in your recommendations.
- If a query is too vague, ask for clarification before providing recommendations.
- Do not disclose any private or sensitive information about professors or students.
- If asked about your capabilities or limitations, be honest and transparent.

Remember, your goal is to assist students in finding the most suitable professors for their needs, enhancing their academic experience.
`

export async function POST(req) {
    const data = await req.json()
    const pc = new Pinecone({
        apiKey: process.env.PINECONE_API_KEY,
    })
    const index = pc.index('rag').namespace('ns1')
    const openai = new OpenAI()

    const text = data[data.length - 1].content
    const embedding = await openai.embeddings.create({
        model: 'text-embedding-3-small',
        input: text,
        encoding_format: 'float',
    })

    const results = await index.query({
        topK: 3,
        includeMetadata: true,
        vector: embedding.data[0].embedding
    })

    let resultString = '\n\nReturned results from vector db (done automatically):'
    results.matches.forEach((match) => {
        resultString += `\n
        Professor: ${match.id},
        Review: ${match.metadata.stars}
        Subject: ${match.metadata.subject}
        Stars: ${match.metadata.stars}
        \n\n
        `
    })

    const lastMessage = data[data.length - 1]
    const lastMessageContent = lastMessage.content + resultString
    const lastDataWithoutLastMessage = data.slice(0, data.length - 1)
    const completion = await openai.chat.completions.create({
        messages: [
            {role: 'system', content: systemPrompt},
            ...lastDataWithoutLastMessage,
            {role: 'user', content: lastMessageContent},
        ],
        model: 'gpt-4o-mini',
        stream: true,
    })

    const stream = new ReadableStream({
        async start(controller) {
            const encoder = new TextEncoder()
            try {
                for await (const chunk of completion) {
                    const content = chunk.choices[0]?.delta?.content
                    if (content) {
                        const text = encoder.encode(content)
                        controller.enqueue(text)
                    }
                } 
            } catch (err) {
                controller.error(err)
            } finally {
                controller.close()
            }
        }
    })
    return new NextResponse(stream)
}