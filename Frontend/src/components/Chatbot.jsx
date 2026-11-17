import React, { useState, useEffect, useRef } from 'react';
import { MessageCircle, X, Send, Tractor, Bot, User, Loader2 } from 'lucide-react';

export default function Chatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: "Hi there! I'm your Smart Farmer Assistant. How can I help you today?", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    const userMessage = { id: Date.now(), text: input, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    // Rule-based engine
    setTimeout(() => {
      let botResponse = "I'm not sure I understand that yet. I can help with crop recommendations, plant disease detection, weather, fertiliser, irrigation, the marketplace, government schemes, and account/login questions. Try asking something like \"How do I detect a plant disease?\"";
      const lowerInput = userMessage.text.toLowerCase();

      if (/\b(delete)\b/.test(lowerInput) && (lowerInput.includes('account') || lowerInput.includes('profile'))) {
        botResponse = "To delete your account, click on your Profile icon in the top right corner. Then select the 'Advanced' tab. You will find the 'Delete Account' option in the Danger Zone at the bottom.";
      } else if (lowerInput.includes('market') || lowerInput.includes('sell') || lowerInput.includes('buy') || lowerInput.includes('price') || lowerInput.includes('mandi')) {
        botResponse = "The Marketplace lets farmers list crops for sale and vendors buy them online. Open 'Marketplace' from the menu to browse or create a listing. Note that payments use Razorpay, which needs to be configured by the site owner.";
      } else if (lowerInput.includes('scheme') || lowerInput.includes('subsidy') || lowerInput.includes('government')) {
        botResponse = "Open the 'Schemes' page to explore government agriculture schemes and subsidies, including eligibility and how to apply.";
      } else if (lowerInput.includes('fertilizer') || lowerInput.includes('fertiliser') || lowerInput.includes('npk') || lowerInput.includes('nutrient')) {
        botResponse = "Use the 'Fertilizer Guide' feature to get a crop-specific NPK plan. For accurate rates, combine it with a soil test from your local agriculture office.";
      } else if (lowerInput.includes('irrigation') || lowerInput.includes('water') || lowerInput.includes('drip')) {
        botResponse = "The 'Irrigation' section lists water requirements and the best method (drip, sprinkler, flood) for each crop, along with practical water-saving tips.";
      } else if (lowerInput.includes('calendar') || lowerInput.includes('sowing') || lowerInput.includes('season') || lowerInput.includes('plant when')) {
        botResponse = "The 'Crop Calendar' shows which crops suit each season — spring, summer, autumn and winter — so you can plan sowing and harvest dates.";
      } else if (lowerInput.includes('disease') || lowerInput.includes('sick') || lowerInput.includes('cure') || lowerInput.includes('leaf spot') || lowerInput.includes('blight')) {
        botResponse = "If your plant looks sick, use our 'Disease Detection' tool. Take a clear, well-lit close-up of the affected leaf, upload it, and select the plant type. Our AI will diagnose the likely disease and show treatment and prevention advice.";
      } else if (lowerInput.includes('weather') || lowerInput.includes('rain') || lowerInput.includes('temperature') || lowerInput.includes('forecast')) {
        botResponse = "You can check the local 5-day weather forecast by using our 'Weather Alerts' feature. It uses live Open-Meteo data — no API key required.";
      } else if (lowerInput.includes('crop') || lowerInput.includes('recommendation')) {
        botResponse = "I can help with crop recommendations! Scroll to the Features section and open 'Crop Recommendation'. Enter your city for a weather-based suggestion, or add your soil test values (N, P, K, pH) to use the AI soil model.";
      } else if (lowerInput.includes('password') || lowerInput.includes('login') || lowerInput.includes('sign in') || lowerInput.includes('otp')) {
        botResponse = "To sign in, click 'Login / Sign Up' in the top menu. Create an account with your email, enter the OTP we email you, then log in with your password. You can also use 'Login with Google'.";
      } else if (/\b(hi|hello|hey|namaste|good morning|good evening)\b/.test(lowerInput)) {
        botResponse = "Hello! How can I assist you with your farming today?";
      } else if (lowerInput.includes('thank') || lowerInput.includes('thanks')) {
        botResponse = "You're welcome! Happy farming. 🌱";
      } else if (lowerInput.includes('support') || lowerInput.includes('help') || lowerInput.includes('contact')) {
        botResponse = "If you need human assistance, email our support team at support@smartfarmer.com. They usually respond within 24 hours.";
      }

      setMessages(prev => [...prev, { id: Date.now(), text: botResponse, sender: 'bot' }]);
      setIsTyping(false);
    }, 1000);
  };

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '56px',
          height: '56px',
          borderRadius: '50%',
          background: 'linear-gradient(135deg, #059669, #34d399)',
          color: 'white',
          border: 'none',
          boxShadow: '0 8px 24px rgba(52, 211, 153, 0.4)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'pointer',
          zIndex: 9999,
          transition: 'transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
          transform: isOpen ? 'scale(0)' : 'scale(1)',
        }}
      >
        <img src="/farmer-icon.png" alt="Chat" style={{ width: '46px', height: '46px', borderRadius: '50%', objectFit: 'cover' }} />
      </button>

      {/* Chat Window */}
      <div
        style={{
          position: 'fixed',
          bottom: '24px',
          right: '24px',
          width: '350px',
          height: '500px',
          maxWidth: 'calc(100vw - 48px)',
          maxHeight: 'calc(100vh - 48px)',
          background: 'var(--bg-secondary)',
          borderRadius: '20px',
          boxShadow: '0 12px 40px rgba(0,0,0,0.4)',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          zIndex: 9999,
          transition: 'all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
          transform: isOpen ? 'scale(1) translateY(0)' : 'scale(0) translateY(100px)',
          opacity: isOpen ? 1 : 0,
          pointerEvents: isOpen ? 'auto' : 'none',
          transformOrigin: 'bottom right'
        }}
      >
        {/* Header */}
        <div style={{
          padding: '16px 20px',
          background: 'var(--bg-tertiary)',
          borderBottom: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
              <img src="/farmer-icon.png" alt="Bot" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
            </div>
            <div>
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)', margin: 0 }}>Smart Farmer AI</h3>
              <p style={{ fontSize: '0.75rem', color: '#10b981', margin: 0, display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: '#10b981', display: 'inline-block' }} /> Online
              </p>
            </div>
          </div>
          <button onClick={() => setIsOpen(false)} style={{ background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}>
            <X size={20} />
          </button>
        </div>

        {/* Message History */}
        <div style={{
          flex: 1,
          padding: '20px',
          overflowY: 'auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '16px'
        }}>
          {messages.map((msg) => (
            <div key={msg.id} style={{
              display: 'flex',
              flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row',
              gap: '8px',
              alignItems: 'flex-end'
            }}>
              <div style={{
                width: '28px', height: '28px', borderRadius: '50%', flexShrink: 0,
                background: msg.sender === 'user' ? 'var(--bg-tertiary)' : 'rgba(52, 211, 153, 0.2)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden',
                color: msg.sender === 'user' ? 'var(--text-secondary)' : '#10b981'
              }}>
                {msg.sender === 'user' ? <User size={14} /> : <img src="/farmer-icon.png" alt="Bot" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />}
              </div>
              <div style={{
                background: msg.sender === 'user' ? 'var(--color-primary)' : 'var(--bg-tertiary)',
                color: msg.sender === 'user' ? 'white' : 'var(--text-primary)',
                padding: '10px 14px',
                borderRadius: '16px',
                borderBottomRightRadius: msg.sender === 'user' ? '4px' : '16px',
                borderBottomLeftRadius: msg.sender === 'bot' ? '4px' : '16px',
                fontSize: '0.875rem',
                lineHeight: 1.5,
                maxWidth: '75%',
                boxShadow: '0 2px 5px rgba(0,0,0,0.05)'
              }}>
                {msg.text}
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end' }}>
              <div style={{ width: '28px', height: '28px', borderRadius: '50%', background: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', overflow: 'hidden' }}>
                <img src="/farmer-icon.png" alt="Bot" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
              </div>
              <div style={{ background: 'var(--bg-tertiary)', padding: '10px 14px', borderRadius: '16px', borderBottomLeftRadius: '4px', display: 'flex', gap: '4px' }}>
                <Loader2 size={16} className="animate-spin" color="var(--text-secondary)" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div style={{
          padding: '16px',
          borderTop: '1px solid var(--border-subtle)',
          background: 'var(--bg-tertiary)'
        }}>
          <form onSubmit={handleSend} style={{ display: 'flex', gap: '8px' }}>
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything..."
              style={{
                flex: 1,
                padding: '10px 16px',
                borderRadius: '20px',
                border: '1px solid var(--border-subtle)',
                background: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
                fontSize: '0.875rem',
                outline: 'none'
              }}
            />
            <button
              type="submit"
              disabled={!input.trim()}
              style={{
                width: '40px', height: '40px',
                borderRadius: '50%',
                background: input.trim() ? 'var(--color-primary)' : 'var(--bg-secondary)',
                color: input.trim() ? 'white' : 'var(--text-muted)',
                border: 'none',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                cursor: input.trim() ? 'pointer' : 'not-allowed',
                transition: 'all 0.2s'
              }}
            >
              <Send size={18} style={{ marginLeft: '-2px' }} />
            </button>
          </form>
        </div>
      </div>
    </>
  );
}
