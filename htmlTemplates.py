css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
}
.chat-message.user {
    background-color: #FFC72C;
}
.chat-message.bot {
    background-color: #DA291C;
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 100px;
  max-height: 100px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.thought-process {
    background-color: #f0f0f0;
    color: #000;
    padding: 10px;
    border-radius: 5px;
    margin-top: 10px;
}
.chat-container {
    height: 400px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="Verbi\static\chatbot image.png" style="max-height: 100px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">
        {{MSG}}
        <details>
            <summary>View thought process</summary>
            <div class="thought-process">
                <pre>{{THOUGHT_PROCESS}}</pre>
            </div>
        </details>
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="Verbi\static\york.png" style="max-height: 100px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''