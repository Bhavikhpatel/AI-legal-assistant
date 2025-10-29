document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const analyzeBtn = document.getElementById('analyzeBtn');
    const queryInput = document.getElementById('queryInput');
    const logsSection = document.getElementById('logsSection');
    const logsBox = document.getElementById('logsBox');
    const resultsSection = document.getElementById('resultsSection');
    const matchedNodeBox = document.getElementById('matchedNodeBox');
    const contextBox = document.getElementById('contextBox');
    const answerBox = document.getElementById('answerBox');
    
    // BNS Modal Elements
    const viewBnsBtn = document.getElementById('viewBnsBtn');
    const bnsModal = document.getElementById('bnsModal');
    const closeModal = document.getElementById('closeModal');

    // Open BNS Modal
    viewBnsBtn.addEventListener('click', function() {
        bnsModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    });

    // Close BNS Modal
    closeModal.addEventListener('click', function() {
        bnsModal.classList.remove('active');
        document.body.style.overflow = 'auto';
    });

    // Close modal when clicking outside
    bnsModal.addEventListener('click', function(e) {
        if (e.target === bnsModal) {
            bnsModal.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && bnsModal.classList.contains('active')) {
            bnsModal.classList.remove('active');
            document.body.style.overflow = 'auto';
        }
    });

    // Analyze Query Button
    analyzeBtn.addEventListener('click', async function() {
        const query = queryInput.value.trim();
        
        if (!query) {
            alert('Please enter a legal query');
            return;
        }

        // Reset UI
        logsBox.innerHTML = '';
        logsSection.style.display = 'block';
        resultsSection.style.display = 'none';
        matchedNodeBox.style.display = 'none';
        contextBox.style.display = 'none';
        answerBox.style.display = 'none';

        // Disable button
        analyzeBtn.disabled = true;
        analyzeBtn.querySelector('.btn-text').textContent = 'Analyzing...';
        analyzeBtn.querySelector('.loader').style.display = 'block';

        try {
            // Call streaming API
            const response = await fetch('/api/analyze-stream', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            handleStreamEvent(data);
                        } catch (e) {
                            console.error('Parse error:', e);
                        }
                    }
                }
            }
        } catch (error) {
            addLog('Error: ' + error.message, 'error');
        } finally {
            // Re-enable button
            analyzeBtn.disabled = false;
            analyzeBtn.querySelector('.btn-text').textContent = 'Analyze Query';
            analyzeBtn.querySelector('.loader').style.display = 'none';
        }
    });

    function handleStreamEvent(data) {
        switch (data.type) {
            case 'log':
                addLog(data.message);
                break;
            
            case 'matched_node':
                resultsSection.style.display = 'flex';
                matchedNodeBox.style.display = 'block';
                document.getElementById('nodeName').textContent = data.node_name;
                const confidence = data.confidence_level || 'MEDIUM';
                const score = (data.similarity_score * 100).toFixed(1);
                document.getElementById('similarityScore').textContent = `${score}% (${confidence})`;
                break;
            
            case 'context':
                contextBox.style.display = 'block';
                document.getElementById('contextContent').textContent = data.context;
                break;
            
            case 'answer':
                answerBox.style.display = 'block';
                // Use simple markdown parsing
                const answerHtml = parseMarkdown(data.answer);
                document.getElementById('answerContent').innerHTML = answerHtml;
                break;
            
            case 'error':
                addLog('Error: ' + data.message, 'error');
                break;
        }
    }

    function addLog(message, type = 'info') {
        const logItem = document.createElement('div');
        logItem.className = `log-item ${type}`;
        logItem.textContent = message;
        logsBox.appendChild(logItem);
        logsBox.scrollTop = logsBox.scrollHeight;
    }

    function parseMarkdown(text) {
        // Simple markdown parser
        let html = text;
        
        // Headers
        html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^\*\*(.+?)\*\*$/gm, '<strong>$1</strong>');
        
        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Lists
        html = html.replace(/^• (.+)$/gm, '<li>$1</li>');
        html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
        
        // Wrap lists
        html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
        
        // Paragraphs
        html = html.replace(/\n\n/g, '</p><p>');
        html = '<p>' + html + '</p>';
        
        // Clean up
        html = html.replace(/<p><\/p>/g, '');
        html = html.replace(/<p>(<h[34]>)/g, '$1');
        html = html.replace(/(<\/h[34]>)<\/p>/g, '$1');
        
        return html;
    }
});
