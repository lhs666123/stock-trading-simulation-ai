"""
Agent Stock Trading Simulation System - Enhanced Iterative Learning Edition (Fixed Trading Frequency)
Three types of agents: Emotional Investor vs. Rational Fund Manager vs. Insider Trader
Added: Reinforcement Learning, Experience Memory, Strategy Optimization, Adaptive Capability
"""

import random
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class StockDataGenerator:
    """Stock Data Generator"""
    
    def __init__(self):
        self.stocks = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
            "META", "NVDA", "NFLX", "AMD", "INTC"
        ]
        self.current_date = datetime(2024, 1, 1)
    
    def generate_stock_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate one month of stock data"""
        stock_data = {}
        
        for stock in self.stocks:
            dates = []
            morning_open = []
            morning_close = []
            afternoon_open = []
            afternoon_close = []
            volumes = []
            highs = []
            lows = []
            
            # Initial price
            base_price = random.uniform(50, 500)
            
            for day in range(days):
                current_date = self.current_date + timedelta(days=day)
                dates.append(current_date)
                
                # Morning trading
                am_open = base_price
                am_volatility = random.uniform(0.01, 0.03)
                am_close = am_open * (1 + random.gauss(0, am_volatility))
                
                # Afternoon trading (based on morning close)
                pm_open = am_close
                pm_volatility = random.uniform(0.008, 0.025)
                pm_close = pm_open * (1 + random.gauss(0, pm_volatility))
                
                # Calculate daily highs and lows
                day_high = max(am_open, am_close, pm_open, pm_close)
                day_low = min(am_open, am_close, pm_open, pm_close)
                
                # Trading volume
                volume = random.randint(1000000, 50000000)
                
                morning_open.append(round(am_open, 2))
                morning_close.append(round(am_close, 2))
                afternoon_open.append(round(pm_open, 2))
                afternoon_close.append(round(pm_close, 2))
                volumes.append(volume)
                highs.append(round(day_high, 2))
                lows.append(round(day_low, 2))
                
                # Update base price
                base_price = pm_close
            
            df = pd.DataFrame({
                'date': dates,
                'morning_open': morning_open,
                'morning_close': morning_close,
                'afternoon_open': afternoon_open,
                'afternoon_close': afternoon_close,
                'volume': volumes,
                'high': highs,
                'low': lows
            })
            
            stock_data[stock] = df
        
        return stock_data
    
    def save_stock_data(self, data: Dict[str, pd.DataFrame], filename: str = "stock_database.json"):
        """Save stock data to JSON file"""
        serializable_data = {}
        
        for stock, df in data.items():
            serializable_data[stock] = {
                'columns': df.columns.tolist(),
                'data': df.to_dict('records')
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Stock data saved to {filename}")


class ReinforcementLearningSystem:
    """Reinforcement Learning System - Enables agents to learn from experience"""
    
    def __init__(self):
        self.q_table = {}  # Q-learning table
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.5  # Increased exploration rate
        self.exploration_decay = 0.99
        
    def get_state_key(self, state_features: Dict) -> str:
        """Convert state features to state key"""
        return json.dumps(state_features, sort_keys=True)
    
    def choose_action(self, state_features: Dict, available_actions: List[str]) -> str:
        """Choose action based on current state"""
        state_key = self.get_state_key(state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        # Exploration-exploitation balance
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        else:
            # If no learning data, choose randomly
            if not self.q_table[state_key]:
                return random.choice(available_actions)
            return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state_features: Dict, action: str, reward: float, next_state_features: Dict):
        """Update Q-value"""
        state_key = self.get_state_key(state_features)
        next_state_key = self.get_state_key(next_state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if next_state_key not in self.q_table:
            next_state_max_q = 0.0
        else:
            next_state_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        current_q = self.q_table[state_key].get(action, 0.0)
        
        # Q-learning update formula
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_state_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
    
    def save_model(self, filename: str = "rl_model.json"):
        """Save reinforcement learning model"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.q_table, f, indent=2, ensure_ascii=False)
        print(f"💾 Reinforcement learning model saved to {filename}")
    
    def load_model(self, filename: str = "rl_model.json"):
        """Load reinforcement learning model"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.q_table = json.load(f)
            print(f"📖 Reinforcement learning model loaded")
        except FileNotFoundError:
            print(f"⚠️  Model file not found, using new model")


class TradingStrategyOptimizer:
    """Trading Strategy Optimizer - Dynamically adjusts trading strategies"""
    
    def __init__(self):
        self.strategy_performance = {}
        self.best_strategies = {}
        self.adaptation_rate = 0.2
        
    def record_strategy_performance(self, strategy_name: str, performance: float):
        """Record strategy performance"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(performance)
        
        # Keep only the last 20 records
        if len(self.strategy_performance[strategy_name]) > 20:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-20:]
    
    def get_best_strategy(self, context: str) -> str:
        """Get best strategy based on context"""
        if not self.strategy_performance:
            return "default"
        
        if context in self.best_strategies:
            # If best strategy for this context is cached, return it directly
            return self.best_strategies[context]
        
        # Calculate average performance of all strategies
        strategy_scores = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                # Use exponential weighted average, recent performance is more important
                weights = np.exp(np.linspace(0, 1, len(performances)))
                weighted_avg = np.average(performances, weights=weights)
                strategy_scores[strategy] = weighted_avg
        
        if not strategy_scores:
            return "default"
        
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        self.best_strategies[context] = best_strategy
        
        return best_strategy
    
    def generate_strategy_variation(self, base_strategy: str, creativity: float = 0.3) -> str:
        """Generate strategy variation"""
        variations = {
            "Conservative": ["Reduce risk preference", "Increase stop-loss points", "Reduce position size"],
            "Aggressive": ["Increase risk preference", "Loosen stop-loss", "Increase position size"],
            "Technical": ["More technical indicators", "Focus on trading volume", "Analyze market structure"],
            "Sentimental": ["Monitor market sentiment", "Track news events", "Follow social media"]
        }
        
        if random.random() < creativity:
            variation_type = random.choice(list(variations.keys()))
            variation = random.choice(variations[variation_type])
            return f"{base_strategy} + {variation}"
        
        return base_strategy


class MarketPatternRecognizer:
    """Market Pattern Recognizer - Learns to recognize market patterns"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_history = []
        
    def analyze_price_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """Analyze price patterns"""
        if len(prices) < 5:
            return {"pattern": "unknown", "confidence": 0.0}
        
        # Calculate technical indicators
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # Identify patterns
        patterns = []
        
        # Trend judgment
        if len(prices) >= 3:
            short_trend = (prices[-1] - prices[-3]) / prices[-3]
            if abs(short_trend) > 0.01:  # Lower threshold
                trend_type = "uptrend" if short_trend > 0 else "downtrend"
                patterns.append({"name": f"Short-term {trend_type}", "confidence": min(0.8, abs(short_trend) * 2)})
        
        # Volatility judgment
        if volatility > 0.01:  # Lower threshold
            patterns.append({"name": "High volatility", "confidence": min(0.9, volatility * 10)})
        elif volatility < 0.008:
            patterns.append({"name": "Low volatility", "confidence": min(0.9, (0.01 - volatility) * 100)})
        
        # Momentum judgment
        if abs(momentum) > 0.03:  # Lower threshold
            momentum_type = "Strong upward" if momentum > 0 else "Strong downward"
            patterns.append({"name": momentum_type, "confidence": min(0.85, abs(momentum) * 3)})
        
        # Return the strongest pattern
        if patterns:
            strongest_pattern = max(patterns, key=lambda x: x["confidence"])
            return strongest_pattern
        else:
            return {"pattern": "No clear pattern", "confidence": 0.5}
    
    def learn_from_pattern(self, pattern: str, outcome: float):
        """Learn from pattern outcomes"""
        if pattern not in self.patterns:
            self.patterns[pattern] = {"outcomes": [], "success_rate": 0.0}
        
        self.patterns[pattern]["outcomes"].append(outcome)
        
        # Calculate success rate
        if len(self.patterns[pattern]["outcomes"]) > 0:
            success_count = sum(1 for o in self.patterns[pattern]["outcomes"] if o > 0)
            self.patterns[pattern]["success_rate"] = success_count / len(self.patterns[pattern]["outcomes"])
        
        # Record history
        self.pattern_history.append({
            "pattern": pattern,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_pattern_advice(self, pattern: str) -> str:
        """Get trading advice based on pattern"""
        if pattern in self.patterns:
            success_rate = self.patterns[pattern]["success_rate"]
            if success_rate > 0.6:
                return f"This pattern has a historical success rate of {success_rate:.1%}, suggesting aggressive trading"
            elif success_rate < 0.4:
                return f"This pattern has a historical success rate of {success_rate:.1%}, suggesting cautious operation"
            else:
                return f"This pattern has a historical success rate of {success_rate:.1%}, suggesting observation or small positions"
        
        return "New pattern, recommend observing before making decisions"


class AIClient:
    """DeepSeek API Client"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        self.conversation_history = []  # Save conversation history
    
    def generate_response(self, system_prompt: str, user_prompt: str, use_history: bool = True) -> str:
        """Generate AI response - Supports conversation history"""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add historical dialogue (if enabled)
            if use_history and self.conversation_history:
                # Keep only the last 5 rounds of dialogue
                recent_history = self.conversation_history[-10:]
                messages.extend(recent_history)
            
            messages.append({"role": "user", "content": user_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.8,
                max_tokens=1200
            )
            
            response_content = response.choices[0].message.content
            
            # Save to conversation history
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return response_content
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return f"Simulated response: Default response from {system_prompt.split('You are')[1].split(',')[0]}"


class BaseTrader:
    """Base Trader Class - Enhanced Iterative Learning Capability"""
    
    def __init__(self, trader_id: int, name: str, initial_capital: float = 100000):
        self.trader_id = trader_id
        self.name = name
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.portfolio = {}  # {stock: shares}
        self.transaction_history = []
        self.weekly_returns = []
        self.weekly_portfolio_values = []
        self.total_return = 0.0
        self.ai_client = AIClient()
        self.memory = []
        self.trading_strategy = ""
        self.personality_traits = self._generate_personality()
        
        # Added iterative learning components
        self.rl_system = ReinforcementLearningSystem()
        self.strategy_optimizer = TradingStrategyOptimizer()
        self.pattern_recognizer = MarketPatternRecognizer()
        self.learning_progress = 0.0  # Learning progress 0-1
        self.adaptation_speed = random.uniform(0.1, 0.3)  # Adaptation speed
        
        # Agent metacognition
        self.meta_cognition = {
            "strengths": [],
            "weaknesses": [],
            "lessons_learned": [],
            "adaptive_changes": []
        }
    
    def _generate_personality(self) -> Dict[str, Any]:
        """Generate trader personality traits"""
        return {
            "confidence": random.uniform(0.3, 0.9),
            "risk_tolerance": random.uniform(0.4, 0.8),
            "talkativeness": random.uniform(0.5, 0.9),
            "analytical": random.uniform(0.3, 0.8),
            "learning_capacity": random.uniform(0.5, 0.9),  # Learning ability
            "trade_frequency": random.uniform(0.3, 0.7)  # Trading frequency tendency
        }
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        stock_value = sum(shares * current_prices.get(stock, 0) 
                         for stock, shares in self.portfolio.items())
        return self.cash + stock_value
    
    def calculate_weekly_return(self, current_prices: Dict[str, float]) -> float:
        """Calculate weekly return - Fixed version"""
        current_value = self.calculate_portfolio_value(current_prices)
        
        if len(self.weekly_portfolio_values) == 0:
            previous_value = self.initial_capital
        else:
            previous_value = self.weekly_portfolio_values[-1]
        
        weekly_return = (current_value - previous_value) / previous_value
        self.weekly_returns.append(weekly_return)
        self.weekly_portfolio_values.append(current_value)
        
        return weekly_return
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary - Fixed version"""
        current_value = self.calculate_portfolio_value({})
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        return {
            "name": self.name,
            "total_return": total_return,
            "weekly_returns": self.weekly_returns,
            "final_cash": self.cash,
            "portfolio": self.portfolio,
            "current_portfolio_value": current_value,
            "initial_capital": self.initial_capital,
            "learning_progress": self.learning_progress
        }
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """Analyze trade outcome and learn"""
        # Extract trading features
        state_features = {
            "stock": trade_decision.get("stock", ""),
            "action": trade_decision.get("action", ""),
            "market_condition": "unknown",
            "trader_confidence": self.personality_traits["confidence"]
        }
        
        # Define available actions
        available_actions = ["buy_small", "buy_medium", "buy_large", 
                           "sell_small", "sell_medium", "sell_large", "hold"]
        
        # Get actual action
        action_type = f"{trade_decision.get('action', 'hold')}_{self._get_size_category(trade_decision.get('shares', 0))}"
        
        # Reward: positive outcome = positive reward, negative outcome = negative reward
        reward = outcome * 100  # Amplify reward
        
        # Update reinforcement learning model
        self.rl_system.update_q_value(state_features, action_type, reward, state_features)
        
        # Record strategy performance
        strategy_name = f"{self.name}_{trade_decision.get('action', 'hold')}_{trade_decision.get('stock', '')}"
        self.strategy_optimizer.record_strategy_performance(strategy_name, outcome)
        
        # Update learning progress
        self.learning_progress = min(1.0, self.learning_progress + abs(outcome) * self.adaptation_speed)
    
    def _get_size_category(self, shares: int) -> str:
        """Determine trading size based on number of shares"""
        if shares <= 3:
            return "small"
        elif shares <= 8:
            return "medium"
        else:
            return "large"
    
    def generate_market_commentary(self, market_data: Dict) -> str:
        """Generate market commentary - Enhanced version"""
        system_prompt = f"""You are a {self.name}, please comment on the current market situation.
Show your personality traits: {self.personality_traits}
You have learned a lot from past trades, and your thinking is now deeper.
Use your accumulated experience to provide unique market analysis."""
        
        user_prompt = f"""Please comment on the current market:

Market overview: {market_data.get('summary', 'No data available')}
Your holdings: {self.portfolio}
Your return rate: {self.get_performance_summary()['total_return']:.2%}
Your learning progress: {self.learning_progress:.1%}

Please share your market views in about 200 words, specifically showing what you've learned from experience:"""
        
        commentary = self.ai_client.generate_response(system_prompt, user_prompt)
        return commentary

    def discuss_strategy_with(self, other_trader: 'BaseTrader', topic: str) -> str:
        """Discuss strategy with other traders - Enhanced version"""
        system_prompt = f"""You are a {self.name}, discussing {topic} with {other_trader.name}.
You are an experienced trader who can learn from other traders' experiences.
Show the wisdom and insight you've gained through iterative learning."""
        
        user_prompt = f"""Please discuss {topic} with {other_trader.name}:

Your strategy: {self.trading_strategy}
Your recent learnings: {self._get_recent_lessons()}
Counterparty type: {other_trader.name}

Please have an in-depth strategy discussion, specifically sharing lessons you've learned from mistakes:"""
        
        discussion = self.ai_client.generate_response(system_prompt, user_prompt)
        return discussion

    def _get_recent_lessons(self) -> str:
        """Get recent learnings"""
        if self.meta_cognition["lessons_learned"]:
            recent_lessons = self.meta_cognition["lessons_learned"][-3:]
            return "\n".join(recent_lessons)
        return "No recent learnings"

    def react_to_news(self, news: str) -> str:
        """React to market news - Enhanced version"""
        system_prompt = f"""You are a {self.name}, please react to the following market news.
Based on your trading experience and learning achievements, provide a rational response.
Show the judgment you've gained through iterative learning."""
        
        user_prompt = f"""News: {news}

Your holdings: {self.portfolio}
Your strategy: {self.trading_strategy}
Your learning progress: {self.learning_progress:.1%}

Please share your views and possible actions, explaining your reasoning process:"""
        
        reaction = self.ai_client.generate_response(system_prompt, user_prompt)
        return reaction

    def share_experience(self) -> str:
        """Share trading experience - Enhanced version"""
        performance = self.get_performance_summary()
        
        system_prompt = f"""You are a {self.name}, please share your trading experience and insights this week.
You are an evolving trader who can learn from every trade.
Describe your growth journey and cognitive evolution in detail."""
        
        user_prompt = f"""Please share your experience as a {self.name}:

【Actual Performance Data】
This week's return: {(self.weekly_returns[-1] if self.weekly_returns else 0):.2%}
Total return: {performance['total_return']:.2%}
Current portfolio value: {performance['current_portfolio_value']:.2f}
Initial capital: {self.initial_capital:.2f}
Current holdings: {self.portfolio}
Learning progress: {self.learning_progress:.1%}

【Iterative Learning Results】
My growth journey: {self._describe_growth()}
Mistakes I've corrected: {self._describe_mistakes()}
The most important lessons I've learned: {self._describe_lessons()}

Please share your mindset, learning process, and self-improvement in detail:"""
        
        experience = self.ai_client.generate_response(system_prompt, user_prompt)
        
        # Record to memory
        self.memory.append({
            "type": "experience_share",
            "content": experience,
            "week": len(self.weekly_returns),
            "timestamp": datetime.now().isoformat(),
            "learning_progress": self.learning_progress
        })
        
        # Update metacognition
        self._update_meta_cognition(experience)
        
        return experience
    
    def _describe_growth(self) -> str:
        """Describe growth journey"""
        if self.learning_progress > 0.7:
            return "I've grown from a novice to an experienced trader, learning to control emotions and risks"
        elif self.learning_progress > 0.4:
            return "I'm learning quickly, gradually understanding market patterns"
        else:
            return "I'm still in the exploration stage, accumulating experience"
    
    def _describe_mistakes(self) -> str:
        """Describe corrected mistakes"""
        mistakes = ["Chasing rallies and selling on dips", "Emotional trading", "Ignoring risk management", "Overconfidence"]
        if self.learning_progress > 0.5:
            learned_mistakes = random.sample(mistakes, 2)
            return f"I've corrected: {', '.join(learned_mistakes)}"
        elif self.learning_progress > 0.2:
            return f"I'm correcting: {random.choice(mistakes)}"
        else:
            return "I'm still making various mistakes and need more learning"
    
    def _describe_lessons(self) -> str:
        """Describe learned lessons"""
        lessons = [
            "Risk management is more important than returns",
            "Emotions are the biggest enemy in trading",
            "Patience in waiting for the best timing",
            "Diversification reduces risk",
            "Learning from mistakes leads to progress"
        ]
        return random.choice(lessons)
    
    def _update_meta_cognition(self, experience: str):
        """Update metacognition"""
        # Extract keywords from experience as learning points
        keywords = ["learned", "understood", "realized", "discovered", "improved", "enhanced"]
        for keyword in keywords:
            if keyword in experience:
                lesson = experience[experience.find(keyword):experience.find(keyword)+100]
                self.meta_cognition["lessons_learned"].append(lesson[:50] + "...")
                break
    
    def learn_from_others(self, others_experiences: List[Dict]) -> str:
        """Learn from others' experiences - Enhanced version"""
        if not others_experiences:
            return "No other traders shared experience this week"
        
        experiences_text = "\n\n".join([
            f"{exp['name']}'s experience:\n{exp['experience']}" 
            for exp in others_experiences
        ])
        
        system_prompt = f"""You are a {self.name}, learning from other traders' experiences.
You are a learning trader who can critically absorb others' experiences.
Combine others' experiences with your own to form deeper understanding."""
        
        user_prompt = f"""Please analyze the following other traders' experience sharing, and engage in deep reflection and integration:

{experiences_text}

Your current strategy: {self.trading_strategy}
Your this week's return: {self.weekly_returns[-1]:.2% if self.weekly_returns else '0%'}
Your learning progress: {self.learning_progress:.1%}

Please explain in detail:
1. What new things have you learned from others' experiences?
2. How will you integrate these experiences into your trading philosophy?
3. Specifically, how will you improve your trading strategy?
4. What specific changes do you plan to make?"""
        
        learning = self.ai_client.generate_response(system_prompt, user_prompt)
        
        # Record learning
        self.memory.append({
            "type": "learning",
            "content": learning,
            "week": len(self.weekly_returns),
            "timestamp": datetime.now().isoformat(),
            "source": "peer_experience"
        })
        
        # Integrate learning into strategy
        enhanced_strategy = self._integrate_learning(learning)
        self.trading_strategy += f"\nWeek {len(self.weekly_returns)} integrated learning: {enhanced_strategy}"
        
        # Update learning progress
        self.learning_progress = min(1.0, self.learning_progress + 0.05)
        
        return learning
    
    def _integrate_learning(self, learning: str) -> str:
        """Integrate learning into strategy"""
        # Extract key learning points
        key_phrases = ["learned", "should", "need", "improve", "adjust", "change"]
        for phrase in key_phrases:
            if phrase in learning:
                start_idx = learning.find(phrase)
                end_idx = min(start_idx + 80, len(learning))
                return learning[start_idx:end_idx]
        
        return learning[:100]
    
    def summarize_final_experience(self) -> str:
        """Summarize final experience - Enhanced version"""
        performance = self.get_performance_summary()
        
        system_prompt = f"""You are a {self.name}, after a month of stock trading, please summarize your final trading experience and evolution journey.
You are an intelligent trader evolving through iterative learning.
Describe your cognitive evolution, strategy improvements, and mindset journey in detail."""
        
        user_prompt = f"""Please summarize your monthly trading experience as a {self.name}:

【Final Performance】
Final return rate: {performance['total_return']:.2%}
Weekly returns: {[f'{r:.2%}' for r in self.weekly_returns]}
Final portfolio value: {performance['current_portfolio_value']:.2f}
Initial capital: {self.initial_capital:.2f}
Final holdings: {self.portfolio}
Final learning progress: {self.learning_progress:.1%}

【Iterative Learning Journey】
Strategy evolution: {self.trading_strategy}
Key learning milestones: {self._get_key_learnings()}
Cognitive evolution: {self._describe_cognitive_evolution()}

Please provide a profound summary, including:
1. Your growth curve
2. The most important cognitive breakthroughs
3. The evolution process of your strategy
4. Thoughts on future trading philosophy"""
        
        final_summary = self.ai_client.generate_response(system_prompt, user_prompt)
        
        self.memory.append({
            "type": "final_summary",
            "content": final_summary,
            "timestamp": datetime.now().isoformat(),
            "final_learning_progress": self.learning_progress
        })
        
        # Save learning models
        self._save_learning_models()
        
        return final_summary
    
    def _get_key_learnings(self) -> str:
        """Get key learnings"""
        key_learnings = []
        for memory_item in self.memory[-5:]:
            if memory_item["type"] in ["learning", "experience_share"]:
                key_learnings.append(memory_item["content"][:50] + "...")
        
        return "\n".join(key_learnings[:3]) if key_learnings else "No records"
    
    def _describe_cognitive_evolution(self) -> str:
        """Describe cognitive evolution"""
        if self.learning_progress > 0.8:
            return "From blind trading to rational analysis, establishing a complete trading system"
        elif self.learning_progress > 0.5:
            return "Starting to understand market patterns, learning emotion management and risk control"
        elif self.learning_progress > 0.3:
            return "Learning from mistakes, gradually forming my own trading method"
        else:
            return "Still in exploration and trial-error stage"
    
    def _save_learning_models(self):
        """Save learning models"""
        trader_folder = f"trader_{self.name}"
        os.makedirs(trader_folder, exist_ok=True)
        
        # Save reinforcement learning model
        self.rl_system.save_model(f"{trader_folder}/rl_model.json")
        
        # Save strategy optimizer
        with open(f"{trader_folder}/strategy_optimizer.json", 'w', encoding='utf-8') as f:
            json.dump({
                "strategy_performance": self.strategy_optimizer.strategy_performance,
                "best_strategies": self.strategy_optimizer.best_strategies
            }, f, indent=2, ensure_ascii=False)
        
        # Save pattern recognizer
        with open(f"{trader_folder}/pattern_recognizer.json", 'w', encoding='utf-8') as f:
            json.dump({
                "patterns": self.pattern_recognizer.patterns,
                "pattern_history": self.pattern_recognizer.pattern_history[-50:]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 {self.name}'s learning models saved")
    
    def load_learning_models(self):
        """Load learning models"""
        trader_folder = f"trader_{self.name}"
        
        # Load reinforcement learning model
        self.rl_system.load_model(f"{trader_folder}/rl_model.json")
        
        # Load strategy optimizer
        try:
            with open(f"{trader_folder}/strategy_optimizer.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.strategy_optimizer.strategy_performance = data.get("strategy_performance", {})
                self.strategy_optimizer.best_strategies = data.get("best_strategies", {})
        except FileNotFoundError:
            pass
        
        # Load pattern recognizer
        try:
            with open(f"{trader_folder}/pattern_recognizer.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.pattern_recognizer.patterns = data.get("patterns", {})
                self.pattern_recognizer.pattern_history = data.get("pattern_history", [])
        except FileNotFoundError:
            pass
        
        print(f"📖 {self.name}'s learning models loaded")


class EmotionalTrader(BaseTrader):
    """Emotional Investor - Enhanced Iterative Learning Capability"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "Emotional Investor")
        self.emotional_state = "neutral"
        self.risk_tolerance = random.uniform(0.6, 0.9)
        self.trading_strategy = "Trading based on market sentiment and price volatility, easily influenced by market emotions"
        self.personality_traits.update({
            "emotional_volatility": random.uniform(0.7, 0.95),
            "herd_mentality": random.uniform(0.6, 0.9),
            "impulsiveness": random.uniform(0.6, 0.9)  # Impulsiveness
        })
        
        # Emotional learning characteristics
        self.emotional_learning = {
            "panic_threshold": random.uniform(0.6, 0.9),
            "fomo_sensitivity": random.uniform(0.5, 0.8),
            "emotional_resilience": 0.5,  # Emotional resilience, improves through learning
            "mistake_memory": []  # Remember emotional mistakes
        }
    
    def make_trading_decisions(self, stock_data: Dict[str, pd.DataFrame], current_day: int) -> List[Dict]:
        """Make trading decisions - Simplified version, increased trading frequency"""
        decisions = []
        
        # Base trading probability, higher on first day
        base_trade_prob = 0.3 if current_day == 0 else 0.2
        
        # Adjust trading probability based on personality
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"]
        
        # Adjust based on learning progress: more trading in early stages to accumulate experience
        if self.learning_progress < 0.3:
            trade_prob *= 1.5
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # Simple random trading decisions
            if random.random() < trade_prob:
                # Decide to buy or sell
                if random.random() < 0.6:  # 60% probability to buy
                    if self.cash > current_price * 10:
                        shares = random.randint(1, 5)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                elif stock in self.portfolio:  # Only sell if holding
                    if random.random() < 0.4:  # 40% probability to sell
                        shares = min(random.randint(1, 3), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """Analyze trade outcome and learn - Special version for Emotional Investor"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        # If emotional mistake, record it
        if outcome < -0.05:
            mistake_record = {
                "stock": trade_decision.get("stock", ""),
                "action": trade_decision.get("action", ""),
                "loss": outcome,
                "timestamp": datetime.now().isoformat()
            }
            self.emotional_learning["mistake_memory"].append(mistake_record)
            
            # Keep only last 10 mistakes
            if len(self.emotional_learning["mistake_memory"]) > 10:
                self.emotional_learning["mistake_memory"] = self.emotional_learning["mistake_memory"][-10:]
        
        # Improve emotional resilience
        if outcome > 0:
            self.emotional_learning["emotional_resilience"] = min(
                0.9, self.emotional_learning["emotional_resilience"] + 0.02
            )


class RationalFundManager(BaseTrader):
    """Rational Fund Manager - Enhanced Iterative Learning Capability"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "Rational Fund Manager")
        self.analysis_depth = random.uniform(0.7, 0.95)
        self.trading_strategy = "Rational investment decisions based on fundamental analysis and technical analysis"
        self.personality_traits.update({
            "analytical": random.uniform(0.8, 0.95),
            "patience": random.uniform(0.7, 0.9),
            "discipline": random.uniform(0.7, 0.9)  # Discipline
        })
        
        # Rational learning characteristics
        self.analytical_models = {
            "trend_model_accuracy": 0.5,
            "pattern_recognition_accuracy": 0.5,
            "risk_model_effectiveness": 0.5,
            "optimization_history": []
        }
    
    def make_trading_decisions(self, stock_data: Dict[str, pd.DataFrame], current_day: int) -> List[Dict]:
        """Make trading decisions - Simplified version, increased trading frequency"""
        decisions = []
        
        # Base trading probability
        base_trade_prob = 0.25
        
        # Adjust based on personality: rational investors trade more cautiously
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"] * 0.8
        
        # More exploration in early learning stages
        if self.learning_progress < 0.4:
            trade_prob *= 1.3
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # Use simple trend analysis
            if current_day >= 5:
                recent_prices = df['afternoon_close'].iloc[current_day-5:current_day+1]
                price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                # Trend trading logic
                if price_change > 0.01 and random.random() < trade_prob:  # Uptrend
                    if self.cash > current_price * 8:
                        shares = random.randint(2, 6)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                
                elif price_change < -0.01 and stock in self.portfolio and random.random() < trade_prob:  # Downtrend
                    shares = min(random.randint(1, 4), self.portfolio[stock])
                    if shares > 0:
                        decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
            
            else:
                # Random exploration in first few days
                if random.random() < trade_prob * 1.5:
                    if self.cash > current_price * 10:
                        shares = random.randint(1, 3)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """Analyze trade outcome and learn - Special version for Rational Fund Manager"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        # Update analytical model accuracy
        if outcome > 0:
            # Successful trade, increase model confidence
            self.analytical_models["trend_model_accuracy"] = min(
                0.95, self.analytical_models["trend_model_accuracy"] + 0.03
            )
        elif outcome < -0.03:
            # Failed trade, slightly decrease confidence
            self.analytical_models["trend_model_accuracy"] = max(
                0.3, self.analytical_models["trend_model_accuracy"] - 0.01
            )


class InformedTrader(BaseTrader):
    """Informed Trader - Enhanced Iterative Learning Capability"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "Informed Trader")
        self.insider_info = {}
        self.trading_strategy = "Trading using informational advantages, knowing certain stock trends in advance"
        self.personality_traits.update({
            "secretive": random.uniform(0.7, 0.9),
            "opportunistic": random.uniform(0.8, 0.95),
            "aggressive": random.uniform(0.6, 0.9)  # Aggressiveness
        })
        
        # Information learning characteristics
        self.info_network = {
            "info_sources": {},
            "info_reliability": 0.7,  # Information reliability
            "timing_accuracy": 0.6,   # Timing accuracy
            "info_history": []        # Information usage history
        }
    
    def set_insider_info(self, stock_data: Dict[str, pd.DataFrame]):
        """Set insider information - Simplified version"""
        # Select 1-2 stocks for insider information
        stocks_with_info = random.sample(list(stock_data.keys()), random.randint(1, 2))
        
        for stock in stocks_with_info:
            df = stock_data[stock]
            if len(df) > 3:
                future_days = random.randint(2, 4)
                direction = random.choice(['up', 'down'])
                strength = random.uniform(0.03, 0.08)
                
                self.insider_info[stock] = {
                    'direction': direction,
                    'strength': strength,
                    'expiry_day': future_days,
                    'confidence': 0.8
                }
    
    def make_trading_decisions(self, stock_data: Dict[str, pd.DataFrame], current_day: int) -> List[Dict]:
        """Make trading decisions - Simplified version, increased trading frequency"""
        decisions = []
        
        # Base trading probability (informed traders are more active)
        base_trade_prob = 0.35
        
        # Adjust based on personality
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"] * 1.2
        
        # More exploration in early learning stages
        if self.learning_progress < 0.5:
            trade_prob *= 1.4
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # 1. First check for insider information
            if stock in self.insider_info:
                info = self.insider_info[stock]
                
                if current_day < info['expiry_day']:
                    if info['direction'] == 'up' and self.cash > current_price * 10:
                        shares = random.randint(3, 8)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                        continue  # Trade with insider information, skip other logic
                    
                    elif info['direction'] == 'down' and stock in self.portfolio:
                        shares = min(random.randint(3, 6), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
                        continue
            
            # 2. Regular trading without insider information
            if random.random() < trade_prob:
                if random.random() < 0.55:  # 55% probability to buy
                    if self.cash > current_price * 12:
                        shares = random.randint(2, 5)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                elif stock in self.portfolio:  # Only sell if holding
                    if random.random() < 0.45:  # 45% probability to sell
                        shares = min(random.randint(2, 4), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """Analyze trade outcome and learn - Special version for Informed Trader"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        stock = trade_decision.get("stock", "")
        
        # Update information reliability
        if stock in self.insider_info:
            info = self.insider_info[stock]
            
            # Check accuracy of insider information
            if outcome > 0 and info['direction'] == 'up':
                # Success, increase information reliability
                self.info_network["info_reliability"] = min(
                    0.95, self.info_network["info_reliability"] + 0.05
                )
            elif outcome < -0.02 and info['direction'] == 'up':
                # Failure, slightly decrease reliability
                self.info_network["info_reliability"] = max(
                    0.3, self.info_network["info_reliability"] - 0.02
                )
            
            # Record information usage history
            info_record = {
                "stock": stock,
                "info_direction": info['direction'],
                "actual_outcome": outcome,
                "expected_strength": info['strength'],
                "reliability_before": self.info_network["info_reliability"],
                "timestamp": datetime.now().isoformat()
            }
            self.info_network["info_history"].append(info_record)
            
            # Keep only recent records
            if len(self.info_network["info_history"]) > 10:
                self.info_network["info_history"] = self.info_network["info_history"][-10:]


class TradingSimulation:
    """Trading Simulation System - Enhanced Iterative Learning Edition"""
    
    def __init__(self):
        self.traders = []
        self.stock_data = {}
        self.conversation_log = []
        self.performance_history = []
        self.market_news = self._generate_market_news()
        self.simulation_round = 0  # Simulation round
        self.cumulative_learning = {}  # Cumulative learning data
    
    def _generate_market_news(self) -> List[Dict]:
        """Generate market news"""
        return [
            {"day": 5, "news": "Federal Reserve announces maintaining interest rates unchanged, market expectations stable"},
            {"day": 12, "news": "Tech stock earnings season approaching, multiple companies exceed performance expectations"},
            {"day": 18, "news": "International oil prices fluctuate significantly, energy sector affected"},
            {"day": 25, "news": "Regulatory policies tighten, some industries face adjustments"}
        ]
    
    def initialize_simulation(self, load_previous_learning: bool = True):
        """Initialize simulation"""
        print("🚀 Initializing iterative learning stock trading simulation system...")
        
        generator = StockDataGenerator()
        self.stock_data = generator.generate_stock_data(30)
        generator.save_stock_data(self.stock_data)
        
        # Create enhanced traders
        self.traders = [
            EmotionalTrader(1),
            RationalFundManager(2),
            InformedTrader(3)
        ]
        
        # Load previous learning (if exists)
        if load_previous_learning:
            for trader in self.traders:
                trader.load_learning_models()
        
        for trader in self.traders:
            if isinstance(trader, InformedTrader):
                trader.set_insider_info(self.stock_data)
        
        print("✅ Iterative learning simulation system initialization completed")
        print(f"📊 Number of stocks: {len(self.stock_data)}")
        print(f"🤖 Traders: {[trader.name for trader in self.traders]}")
        
        # Show initial learning status
        for trader in self.traders:
            print(f"   {trader.name}: Learning progress {trader.learning_progress:.1%}")
    
    def execute_trades(self, decisions: List[Dict], trader: BaseTrader):
        """Execute trades"""
        for decision in decisions:
            stock = decision["stock"]
            action = decision["action"]
            shares = decision["shares"]
            price = decision["price"]
            
            if action == "buy":
                cost = shares * price
                if trader.cash >= cost:
                    trader.cash -= cost
                    trader.portfolio[stock] = trader.portfolio.get(stock, 0) + shares
                    trader.transaction_history.append({
                        "day": len(trader.weekly_returns) * 7,
                        "action": "buy",
                        "stock": stock,
                        "shares": shares,
                        "price": price,
                        "cost": cost
                    })
                    print(f"   ✅ {trader.name} bought {shares} shares of {stock} @ {price:.2f}")
            
            elif action == "sell":
                if trader.portfolio.get(stock, 0) >= shares:
                    revenue = shares * price
                    trader.cash += revenue
                    trader.portfolio[stock] -= shares
                    if trader.portfolio[stock] == 0:
                        del trader.portfolio[stock]
                    trader.transaction_history.append({
                        "day": len(trader.weekly_returns) * 7,
                        "action": "sell",
                        "stock": stock,
                        "shares": shares,
                        "price": price,
                        "revenue": revenue
                    })
                    print(f"   ✅ {trader.name} sold {shares} shares of {stock} @ {price:.2f}")
    
    def analyze_trade_outcomes(self, day_trades: Dict[str, List[Dict]], current_day: int):
        """Analyze trade outcomes and enable agent learning"""
        if current_day == 0:
            return
        
        # Get next day's prices for calculating returns
        if current_day >= len(list(self.stock_data.values())[0]):
            return
        
        next_day_prices = {}
        for stock, df in self.stock_data.items():
            if current_day < len(df) - 1:
                next_day_prices[stock] = df.iloc[current_day + 1]['afternoon_close']
        
        # Analyze each trader's trade outcomes
        for trader_name, trades in day_trades.items():
            trader = next((t for t in self.traders if t.name == trader_name), None)
            if not trader or not trades:
                continue
            
            for trade in trades:
                stock = trade["stock"]
                action = trade["action"]
                price = trade["price"]
                
                if stock in next_day_prices:
                    next_price = next_day_prices[stock]
                    
                    if action == "buy":
                        # Buying profit is next day's price change
                        profit = (next_price - price) / price
                    elif action == "sell":
                        # Selling profit is avoided loss (assuming holding until next day if not sold)
                        profit = (price - next_price) / price  # Note: this is avoided loss
                    else:
                        profit = 0
                    
                    # Enable agent to learn from trade outcomes
                    trader.analyze_trade_outcome(trade, profit)
    
    def run_market_commentary(self, current_day: int):
        """Run market commentary"""
        print(f"\n📢 Day {current_day} market commentary")
        
        market_data = {
            "summary": f"Day {current_day} trading situation",
            "active_stocks": list(self.stock_data.keys())[:3]
        }
        
        # Select traders with highest learning progress to comment
        commentators = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)[:2]
        for trader in commentators:
            if random.random() < trader.personality_traits["talkativeness"]:
                print(f"\n{trader.name} (Learning progress: {trader.learning_progress:.1%}) provides market commentary:")
                commentary = trader.generate_market_commentary(market_data)
                print(f"{trader.name}: {commentary}")
                
                self.conversation_log.append({
                    "day": current_day,
                    "speaker": trader.name,
                    "learning_progress": trader.learning_progress,
                    "type": "market_commentary",
                    "content": commentary,
                    "timestamp": datetime.now().isoformat()
                })
                time.sleep(1)
    
    def run_strategy_discussion(self, week: int):
        """Run strategy discussion"""
        print(f"\n💬 Week {week} in-depth strategy discussion")
        
        discussion_topics = [
            "Experience of learning from mistakes",
            "Key milestones in strategy evolution", 
            "How to balance risk and return",
            "Iterative process of market cognition"
        ]
        
        topic = random.choice(discussion_topics)
        print(f"Discussion topic: {topic}")
        
        # Select two traders with highest learning progress for discussion
        participants = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)[:2]
        trader1, trader2 = participants
        
        print(f"\n{trader1.name} (Learning progress: {trader1.learning_progress:.1%}) initiates discussion:")
        discussion1 = trader1.discuss_strategy_with(trader2, topic)
        print(f"{trader1.name}: {discussion1}")
        
        self.conversation_log.append({
            "week": week,
            "speaker": trader1.name,
            "learning_progress": trader1.learning_progress,
            "type": "strategy_discussion",
            "content": discussion1,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        })
        
        time.sleep(1)
        
        print(f"\n{trader2.name} (Learning progress: {trader2.learning_progress:.1%}) responds:")
        discussion2 = trader2.discuss_strategy_with(trader1, topic)
        print(f"{trader2.name}: {discussion2}")
        
        self.conversation_log.append({
            "week": week,
            "speaker": trader2.name,
            "learning_progress": trader2.learning_progress,
            "type": "strategy_discussion", 
            "content": discussion2,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        })
    
    def run_news_reaction(self, current_day: int):
        """Run news reaction"""
        today_news = [news for news in self.market_news if news["day"] == current_day]
        
        if today_news:
            for news_item in today_news:
                print(f"\n📰 Market news: {news_item['news']}")
                
                # All traders react to news
                for trader in self.traders:
                    print(f"\n{trader.name} (Learning progress: {trader.learning_progress:.1%}) reacts to news:")
                    reaction = trader.react_to_news(news_item['news'])
                    print(f"{trader.name}: {reaction}")
                    
                    self.conversation_log.append({
                        "day": current_day,
                        "speaker": trader.name,
                        "learning_progress": trader.learning_progress,
                        "type": "news_reaction",
                        "content": reaction,
                        "news": news_item['news'],
                        "timestamp": datetime.now().isoformat()
                    })
                    time.sleep(1)
    
    def run_weekly_discussion(self, week: int):
        """Run weekly discussion"""
        print(f"\n🗣️ Week {week} trading experience sharing session")
        
        # Sort by learning progress, let those with most progress share first
        sorted_traders = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)
        
        experiences = []
        for trader in sorted_traders:
            print(f"\n{trader.name} (Learning progress: {trader.learning_progress:.1%}) is sharing experience...")
            experience = trader.share_experience()
            experiences.append({
                "name": trader.name,
                "experience": experience,
                "learning_progress": trader.learning_progress
            })
            
            self.conversation_log.append({
                "week": week,
                "speaker": trader.name,
                "learning_progress": trader.learning_progress,
                "type": "experience_share",
                "content": experience,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"{trader.name}: {experience[:100]}...")
            time.sleep(1)
        
        # Traders learn from each other (especially low learning progress from high)
        print(f"\n🎓 Week {week} mutual learning session")
        
        # Group by learning progress
        high_learners = [t for t in self.traders if t.learning_progress > 0.5]
        low_learners = [t for t in self.traders if t.learning_progress <= 0.5]
        
        for learner in low_learners:
            # Let low learning progress learn from high learning progress
            if high_learners:
                teacher_experiences = [exp for exp in experiences if exp["name"] in [h.name for h in high_learners]]
                if teacher_experiences:
                    print(f"{learner.name} (Learning progress: {learner.learning_progress:.1%}) is learning from experts...")
                    
                    learning = learner.learn_from_others(teacher_experiences)
                    self.conversation_log.append({
                        "week": week,
                        "speaker": learner.name,
                        "learning_progress": learner.learning_progress,
                        "type": "learning",
                        "content": learning,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    print(f"{learner.name}'s learning insights: {learning[:100]}...")
                    time.sleep(1)
    
    def run_simulation(self, rounds: int = 1):
        """Run complete simulation (supports multiple rounds)"""
        for round_num in range(rounds):
            self.simulation_round = round_num + 1
            print(f"\n🎯 Starting round {self.simulation_round} stock trading simulation...")
            
            if round_num > 0:
                # New simulation round, maintain learning state but reset some data
                print("🔄 Starting new simulation round, preserving learning results...")
                for trader in self.traders:
                    # Reset cash and portfolio, but maintain learning models
                    trader.cash = trader.initial_capital
                    trader.portfolio = {}
                    trader.weekly_returns = []
                    trader.weekly_portfolio_values = []
                    trader.transaction_history = []
            
            total_days = 30
            weeks = 4
            days_per_week = total_days // weeks
            
            for week in range(weeks):
                print(f"\n{'='*60}")
                print(f"📅 Week {week+1} trading begins (Round {self.simulation_round})")
                print(f"{'='*60}")
                
                # Weekly trading
                for day_in_week in range(days_per_week):
                    current_day = week * days_per_week + day_in_week
                    
                    print(f"\n--- Day {current_day+1} ---")
                    
                    # Record today's trades
                    day_trades = {}
                    
                    # Market commentary (every few days)
                    if current_day % 3 == 0:
                        self.run_market_commentary(current_day + 1)
                    
                    # News reaction
                    self.run_news_reaction(current_day + 1)
                    
                    # Execute trades
                    trade_count = 0
                    for trader in self.traders:
                        decisions = trader.make_trading_decisions(self.stock_data, current_day)
                        day_trades[trader.name] = decisions
                        
                        if decisions:
                            trade_count += len(decisions)
                            self.execute_trades(decisions, trader)
                    
                    print(f"🤝 Completed {trade_count} trades today")
                    
                    # Analyze trade outcomes and learn
                    self.analyze_trade_outcomes(day_trades, current_day)
                
                # Calculate weekly returns
                current_prices = self._get_week_end_prices(week, days_per_week)
                
                print(f"\n💰 Week {week+1} returns:")
                for trader in self.traders:
                    weekly_return = trader.calculate_weekly_return(current_prices)
                    performance = trader.get_performance_summary()
                    print(f"   {trader.name}: Weekly return {weekly_return:+.2%}, Total return {performance['total_return']:+.2%}, Learning progress {trader.learning_progress:.1%}")
                
                # Strategy discussion (weekly)
                self.run_strategy_discussion(week + 1)
                
                # Weekly discussion and learning
                self.run_weekly_discussion(week + 1)
                
                # Record performance
                self.performance_history.append({
                    "round": self.simulation_round,
                    "week": week + 1,
                    "returns": {trader.name: trader.weekly_returns[-1] for trader in self.traders},
                    "learning_progress": {trader.name: trader.learning_progress for trader in self.traders}
                })
            
            # Final summary
            self.run_final_summary()
    
    def _get_week_end_prices(self, week: int, days_per_week: int) -> Dict[str, float]:
        """Get weekend prices"""
        current_prices = {}
        current_day = (week + 1) * days_per_week - 1
        
        for stock, df in self.stock_data.items():
            if current_day < len(df):
                current_prices[stock] = df.iloc[current_day]['afternoon_close']
        
        return current_prices
    
    def run_final_summary(self):
        """Run final summary"""
        print("\n🎊 Monthly trading simulation completed!")
        print("\n📈 Final performance report:")
        
        final_returns = {}
        learning_progresses = {}
        final_summaries = []
        
        for trader in self.traders:
            performance = trader.get_performance_summary()
            final_return = performance['total_return']
            final_returns[trader.name] = final_return
            learning_progresses[trader.name] = trader.learning_progress
            
            print(f"\n{trader.name}:")
            print(f"  Total return: {final_return:.2%}")
            print(f"  Learning progress: {trader.learning_progress:.1%}")
            print(f"  Final cash: {trader.cash:.2f}")
            print(f"  Final holdings: {trader.portfolio}")
            print(f"  Portfolio value: {performance['current_portfolio_value']:.2f}")
            
            print(f"{trader.name} is summarizing final experience...")
            final_summary = trader.summarize_final_experience()
            final_summaries.append({
                "name": trader.name,
                "summary": final_summary,
                "final_return": final_return,
                "learning_progress": trader.learning_progress
            })
            
            self.conversation_log.append({
                "round": self.simulation_round,
                "speaker": trader.name,
                "learning_progress": trader.learning_progress,
                "type": "final_summary",
                "content": final_summary,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"{trader.name}'s final summary: {final_summary[:100]}...")
            time.sleep(1)
        
        # Best trader (considering both returns and learning)
        combined_scores = {}
        for name in final_returns.keys():
            # Return weight 0.6, learning progress weight 0.4
            return_score = (final_returns[name] + 1) / 2  # Normalized to 0-1
            learning_score = learning_progresses[name]
            combined_score = return_score * 0.6 + learning_score * 0.4
            combined_scores[name] = combined_score
        
        best_trader = max(combined_scores.items(), key=lambda x: x[1])
        print(f"\n🏆 This month's best trader: {best_trader[0]} (Combined score: {best_trader[1]:.2f})")
        
        # Learning progress award
        learning_improvement = {name: learning_progresses[name] for name in learning_progresses}
        most_improved = max(learning_improvement.items(), key=lambda x: x[1])
        print(f"📚 Most learning progress: {most_improved[0]} (Learning progress: {most_improved[1]:.1%})")
        
        # Save learning summary
        self.save_learning_summary(final_summaries)
        
        # Save results
        self.save_results(final_summaries)
    
    def save_learning_summary(self, final_summaries: List[Dict]):
        """Save learning summary"""
        # Update cumulative learning data
        if "rounds" not in self.cumulative_learning:
            self.cumulative_learning["rounds"] = []
        
        round_summary = {
            "round": self.simulation_round,
            "date": datetime.now().isoformat(),
            "traders": {}
        }
        
        for trader in self.traders:
            performance = trader.get_performance_summary()
            round_summary["traders"][trader.name] = {
                "final_return": performance['total_return'],
                "learning_progress": trader.learning_progress,
                "final_portfolio_value": performance['current_portfolio_value']
            }
        
        self.cumulative_learning["rounds"].append(round_summary)
        
        # Save to file
        with open("cumulative_learning.json", "w", encoding="utf-8") as f:
            json.dump(self.cumulative_learning, f, indent=2, ensure_ascii=False)
    
    def save_results(self, final_summaries: List[Dict]):
        """Save results to files"""
        # Save conversation log
        with open(f"trading_conversations_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
        
        # Save agent memory
        memory_data = {}
        for trader in self.traders:
            performance = trader.get_performance_summary()
            memory_data[trader.name] = {
                "final_return": performance['total_return'],
                "learning_progress": trader.learning_progress,
                "current_portfolio_value": performance['current_portfolio_value'],
                "initial_capital": trader.initial_capital,
                "memory": trader.memory[-20:],  # Keep only last 20 memories
                "trading_strategy": trader.trading_strategy,
                "final_portfolio": trader.portfolio,
                "final_cash": trader.cash,
                "weekly_returns": trader.weekly_returns,
                "personality_traits": trader.personality_traits,
                "meta_cognition": trader.meta_cognition
            }
        
        with open(f"trading_experience_memory_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        # Save performance history
        with open(f"trading_performance_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
        
        print("✅ All results saved to JSON files!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Iterative Learning Agent Stock Trading Simulation System')
    parser.add_argument('--days', type=int, default=30, help='Simulation days')
    parser.add_argument('--weeks', type=int, default=4, help='Simulation weeks')
    parser.add_argument('--rounds', type=int, default=1, help='Simulation rounds')
    parser.add_argument('--fast', action='store_true', help='Fast mode (reduce dialogue)')
    parser.add_argument('--reset-learning', action='store_true', help='Reset learning models')
    
    args = parser.parse_args()
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ Please set DEEPSEEK_API_KEY environment variable")
        return
    
    print("🧠 Starting iterative learning stock trading simulation system")
    print("=" * 50)
    print("Special features:")
    print("1. Reinforcement Learning - Agents learn from every trade")
    print("2. Strategy Optimization - Dynamically adjusts trading strategies")
    print("3. Pattern Recognition - Learns to recognize market patterns")
    print("4. Metacognition - Agents understand their strengths and weaknesses")
    print("5. Multi-round Iteration - Agents become smarter over time")
    print("=" * 50)
    
    simulation = TradingSimulation()
    simulation.initialize_simulation(load_previous_learning=not args.reset_learning)
    
    if args.fast:
        print("⚡ Fast mode: Simplified dialogue process")
        # Can add simplified logic here
    
    simulation.run_simulation(rounds=args.rounds)
    
    print("\n🎯 Simulation completed!")
    print("Agents' learning models saved, will continue learning next run")


if __name__ == "__main__":
    main()