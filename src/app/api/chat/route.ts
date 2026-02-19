import { NextRequest, NextResponse } from 'next/server'
import { ChatOpenAI } from '@langchain/openai'
import {
  HumanMessage,
  SystemMessage,
  BaseMessage,
  ToolMessage,
} from '@langchain/core/messages'
import { Neo4jGraph } from '@langchain/community/graphs/neo4j_graph'
import { createPersonSearchTool } from '@/modules/agent/tools/person-search.tool'
import { SYSTEM_PROMPTS } from '@/lib/simulation/system-prompts'
import type { AssistantMode } from '@/lib/simulation'
import { DynamicTool } from '@langchain/core/tools'

// Allow streaming responses up to 30 seconds
export const maxDuration = 30

interface ChatRequest {
  messages: Array<{ role: string; content: string }>
  system?: string
  tools?: Record<string, unknown>
  aiMode?: AssistantMode
}

interface StreamEvent {
  type:
    | 'tool_call'
    | 'tool_result'
    | 'tool_error'
    | 'message'
    | 'done'
    | 'error'
  tool?: string
  args?: unknown
  result?: unknown
  content?: string
  error?: string
}

export async function POST(req: NextRequest) {
  const encoder = new TextEncoder()

  try {
    const { messages, system, aiMode }: ChatRequest = await req.json()

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return NextResponse.json(
        { error: 'Messages array is required' },
        { status: 400 }
      )
    }

    const mode = aiMode || 'default'
    const systemPrompt = system || SYSTEM_PROMPTS[mode]

    // Build streaming response
    const stream = new ReadableStream({
      async start(controller) {
        try {
          // Initialize tools
          const langchainTools = [
            new DynamicTool({
              name: 'search_person',
              description:
                'Search for a person in GoalPost by name. Use this whenever the user asks about someone specific.',
              func: async (name: string) => {
                try {
                  const graph = await Neo4jGraph.initialize({
                    url: process.env.NEO4J_URI!,
                    username: process.env.NEO4J_USERNAME!,
                    password: process.env.NEO4J_PASSWORD!,
                  })

                  const personTool = createPersonSearchTool(graph)
                  const result = await personTool.invoke({ name })
                  return result
                } catch {
                  return JSON.stringify({
                    found: false,
                    message: `Could not find ${name} in GoalPost`,
                  })
                }
              },
            }),

            new DynamicTool({
              name: 'search_community',
              description: 'Search for communities in GoalPost',
              func: async (query: string) => {
                try {
                  const graph = await Neo4jGraph.initialize({
                    url: process.env.NEO4J_URI!,
                    username: process.env.NEO4J_USERNAME!,
                    password: process.env.NEO4J_PASSWORD!,
                  })

                  const cypherQuery = `
                    MATCH (c:Community)
                    WHERE toLower(c.name) CONTAINS toLower($query)
                    RETURN c.name as name, c.description as description
                    LIMIT 5
                  `

                  const results = await graph.query(cypherQuery, { query })
                  return JSON.stringify({
                    found: results.length > 0,
                    communities: results,
                    count: results.length,
                  })
                } catch {
                  return JSON.stringify({ found: false, communities: [] })
                }
              },
            }),
          ]

          // Initialize model
          const model = new ChatOpenAI({
            apiKey: process.env.OPENAI_API_KEY,
            model: 'gpt-4.1',
            temperature: 0.7,
            maxTokens: 2048,
          })

          const modelWithTools = model.bindTools(langchainTools)

          // Convert UI messages to LangChain format
          const messageHistory: BaseMessage[] = [
            new SystemMessage(systemPrompt),
            ...messages.map((msg) => {
              if (msg.role === 'user' || msg.role === 'assistant') {
                return new HumanMessage(msg.content)
              }
              return new HumanMessage(msg.content)
            }),
          ]

          // First invoke
          let response = await modelWithTools.invoke(messageHistory)

          // Handle tool calls
          if (response.tool_calls && response.tool_calls.length > 0) {
            const toolsMap = Object.fromEntries(
              langchainTools.map((t) => [t.name, t])
            )
            const toolResults: ToolMessage[] = []

            // Execute tools
            for (const toolCall of response.tool_calls) {
              try {
                // Send tool_call event
                const toolCallEvent: StreamEvent = {
                  type: 'tool_call',
                  tool: toolCall.name,
                  args: toolCall.args,
                }
                controller.enqueue(
                  encoder.encode(JSON.stringify(toolCallEvent) + '\n')
                )

                const tool = toolsMap[toolCall.name]
                if (!tool) continue

                const result = await tool.invoke(toolCall.args || {})

                // Send tool_result event
                const resultEvent: StreamEvent = {
                  type: 'tool_result',
                  tool: toolCall.name,
                  result: JSON.parse(result as string),
                }
                controller.enqueue(
                  encoder.encode(JSON.stringify(resultEvent) + '\n')
                )

                if (toolCall.id) {
                  toolResults.push(
                    new ToolMessage({
                      content: result as string,
                      tool_call_id: toolCall.id,
                      name: toolCall.name,
                    })
                  )
                }
              } catch (error) {
                // Send tool_error event
                const errorEvent: StreamEvent = {
                  type: 'tool_error',
                  tool: toolCall.name,
                  error:
                    error instanceof Error ? error.message : 'Unknown error',
                }
                controller.enqueue(
                  encoder.encode(JSON.stringify(errorEvent) + '\n')
                )

                if (toolCall.id) {
                  toolResults.push(
                    new ToolMessage({
                      content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
                      tool_call_id: toolCall.id,
                      name: toolCall.name,
                    })
                  )
                }
              }
            }

            // Add results and get final response
            messageHistory.push(response)
            messageHistory.push(...toolResults)
            messageHistory.push(
              new HumanMessage(
                'Based on the tool results above, please provide a clear and engaging response.'
              )
            )

            response = await modelWithTools.invoke(messageHistory)
          }

          // Send message event
          const finalText = (response.content as string) || ''
          const messageEvent: StreamEvent = {
            type: 'message',
            content: finalText,
          }
          controller.enqueue(
            encoder.encode(JSON.stringify(messageEvent) + '\n')
          )

          // Send done event
          const doneEvent: StreamEvent = {
            type: 'done',
          }
          controller.enqueue(encoder.encode(JSON.stringify(doneEvent) + '\n'))
        } catch (error) {
          const errorEvent: StreamEvent = {
            type: 'error',
            error:
              error instanceof Error ? error.message : 'Internal server error',
          }
          controller.enqueue(encoder.encode(JSON.stringify(errorEvent) + '\n'))
        } finally {
          controller.close()
        }
      },
    })

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'application/x-ndjson',
        'Cache-Control': 'no-cache',
        Connection: 'keep-alive',
      },
    })
  } catch (error) {
    const message =
      error instanceof Error ? error.message : 'Internal server error'
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
