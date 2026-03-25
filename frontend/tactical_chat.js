/**
 * Directive 4: Analyst Chat UI Frontend Logic
 * Connects to the MAS WebSocket and injects dynamic timeline linking.
 */

document.addEventListener("DOMContentLoaded", () => {
    const chatInput = document.getElementById("ai-chat-input");
    const sendBtn = document.getElementById("ai-send-btn");
    const messagesContainer = document.getElementById("ai-chat-messages");

    let ws = null;
    let isConnected = false;

    // Auto-connect to Orchestrator upon loading the dashboard page
    setTimeout(() => {
        if (!isConnected) connectWebSocket();
    }, 1000);

    // Connect to specific post-match endpoint
    function connectWebSocket() {
        const wsUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? "ws://127.0.0.1:8000/ws/post_match_chat"
            : `wss://${window.location.hostname}/ws/post_match_chat`;

        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            isConnected = true;
            appendMessage("System", "Connected to Tactical AI Assistant.", "system");
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                const agentName = data.agent || "AI";
                let rawText = data.text || "";

                // Parse markdown into HTML if available
                let parsedHtml = rawText;
                if (window.marked && data.type !== "user" && agentName !== "System") {
                    parsedHtml = marked.parse(rawText);
                }

                // Directive 4: Dynamic Linking
                // Regex looks for "Frame 123" or "[Frame 123]" and makes it clickable
                parsedHtml = parsedHtml.replace(/Frame\s*(\d+)/gi, (match, frameNum) => {
                    return `<span class="text-[#ccff00] cursor-pointer underline decoration-[#ccff00]/50 font-black jump-link hover:bg-[#ccff00] hover:text-black transition-colors" data-frame="${frameNum}">${match}</span>`;
                });

                // Rely on appendMessage to render the template so we don't hit ReferenceErrors!
                appendMessage(agentName, parsedHtml, data.type || "normal");

                // Attach event listeners to new links without replacing nodes
                document.querySelectorAll(".jump-link").forEach(link => {
                    // cloneNode to strip old listeners
                    const newLink = link.cloneNode(true);
                    if (link.parentNode) {
                        link.parentNode.replaceChild(newLink, link);
                        newLink.addEventListener("click", (e) => {
                            const targetFrame = e.target.getAttribute("data-frame");
                            jumpToFrame(targetFrame);
                        });
                    }
                });
            } catch (e) {
                console.error("Error parsing WS message:", e);
            }
        };

        ws.onclose = () => {
            isConnected = false;
            appendMessage("System", "Connection lost. Please refresh.", "system");
        };
    }

    function appendMessage(sender, text, type) {
        if (!messagesContainer) return;

        const msgDiv = document.createElement("div");
        msgDiv.className = "mb-4 text-sm font-mono leading-relaxed border-l-2 pl-4";

        let headerColor = "text-[#ccff00]"; // Default (Coach/You)
        let borderColor = "border-[#ccff00]";
        let textColor = "text-white";

        if (sender === "System") {
            headerColor = "text-[#8a8a8a]";
            borderColor = "border-[#8a8a8a]";
            textColor = "text-[#8a8a8a]";
        }
        if (sender === "Orchestrator") {
            headerColor = "text-[#ff2a4d]";
            borderColor = "border-[#ff2a4d]";
            textColor = "text-white";
        }
        if (sender === "Scout") headerColor = "text-[#ccff00]";
        if (sender === "Judge") headerColor = "text-[#ff2a4d]";

        const prefix = type === "thinking" ? "<span class='animate-pulse mr-2 text-[#ccff00]'>[SYNC]</span>" : "";

        // Add styling border dynamically
        msgDiv.classList.add(borderColor);

        msgDiv.innerHTML = `
            <div class="font-bold font-oswald flex items-center gap-2 ${headerColor} mb-1 uppercase tracking-widest text-xs">
                ${prefix} ${sender}
            </div>
            <div class="${textColor} ${type === 'thinking' ? 'italic opacity-70' : ''} prose prose-invert prose-sm max-w-none prose-p:leading-snug prose-li:my-0 prose-a:text-[#ccff00] prose-strong:text-white">${text}</div>
        `;

        messagesContainer.appendChild(msgDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    function sendMessage() {
        const text = chatInput.value.trim();
        if (text && ws && ws.readyState === WebSocket.OPEN) {
            appendMessage("Coach (You)", text, "user");
            ws.send(text);
            chatInput.value = "";
        }
    }

    if (sendBtn) {
        sendBtn.addEventListener("click", sendMessage);
    }

    if (chatInput) {
        chatInput.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
    }

    /**
     * Integrates with the massive Three.js digital twin.
     * Scrubbing the slider triggers the `fetchTacticalData()` in tactical_engine.js
     * which implicitly updates the 3D meshes.
     */
    function jumpToFrame(frameNum) {
        const slider = document.getElementById("frame-slider");
        if (slider) {
            slider.value = frameNum;
            // Dispatch a change event so the main app picks it up natively
            const event = new Event("change");
            slider.dispatchEvent(event);
            console.log(`[AI Navigator] Jumped timeline to Frame ${frameNum}`);
        } else {
            console.warn("Frame slider not found on page.");
        }
    }
});
