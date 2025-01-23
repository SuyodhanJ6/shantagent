openapi: 3.0.0
info:
  title: Shantagent API
  version: 0.1.0
  description: Multi-agent service supporting chat and ReAct capabilities

servers:
  - url: http://localhost:8000/v1
    description: Local development server

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer

  schemas:
    ChatMessage:
      type: object
      properties:
        type:
          type: string
          enum: [human, ai]
          description: Role of the message
        content:
          type: string
          description: Content of the message
        metadata:
          type: object
          description: Additional message metadata
          
    UserInput:
      type: object
      required:
        - message
      properties:
        message:
          type: string
          description: User input to the agent
        model:
          type: string
          description: Model to use for generation
          default: mixtral-8x7b-32768
        stream:
          type: boolean
          description: Whether to stream the response
          default: false
        thread_id:
          type: string
          description: Thread ID for conversation
        metadata:
          type: object
          description: Additional metadata

    ChatHistory:
      type: object
      properties:
        messages:
          type: array
          items:
            $ref: '#/components/schemas/ChatMessage'
        thread_id:
          type: string
        metadata:
          type: object

paths:
  /chat:
    post:
      summary: Basic chat interaction
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessage'

  /chat/stream:
    post:
      summary: Stream chat responses
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: Streaming response
          content:
            text/event-stream:
              schema:
                type: string

  /chat/history/{thread_id}:
    get:
      summary: Get chat history
      parameters:
        - name: thread_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Chat history
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatHistory'

  /research:
    post:
      summary: Research-based chat
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: Research response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessage'

  /research/stream:
    post:
      summary: Stream research responses
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: Streaming research response
          content:
            text/event-stream:
              schema:
                type: string

  /background-task:
    post:
      summary: Process background tasks
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UserInput'
      responses:
        '200':
          description: Task response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ChatMessage'

  /health:
    get:
      summary: System health check
      responses:
        '200':
          description: Health status
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                  components:
                    type: object

  /metrics:
    get:
      summary: Prometheus metrics
      responses:
        '200':
          description: Metrics data
          content:
            text/plain:
              schema:
                type: string

security:
  - BearerAuth: []

tags:
  - name: chat
    description: Chat endpoints
  - name: research
    description: Research endpoints
  - name: background-task
    description: Background task endpoints
  - name: system
    description: System endpoints