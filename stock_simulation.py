"""
智能体股票交易模拟系统 - 迭代学习增强版（修复交易频率）
三类智能体：情绪投资者 vs 理性基金经理 vs 信息泄露者
新增：强化学习、经验记忆、策略优化、自适应能力
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

# 加载环境变量
load_dotenv()


class StockDataGenerator:
    """股票数据生成器"""
    
    def __init__(self):
        self.stocks = [
            "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA",
            "META", "NVDA", "NFLX", "AMD", "INTC"
        ]
        self.current_date = datetime(2024, 1, 1)
    
    def generate_stock_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """生成一个月的股票数据"""
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
            
            # 初始价格
            base_price = random.uniform(50, 500)
            
            for day in range(days):
                current_date = self.current_date + timedelta(days=day)
                dates.append(current_date)
                
                # 上午交易
                am_open = base_price
                am_volatility = random.uniform(0.01, 0.03)
                am_close = am_open * (1 + random.gauss(0, am_volatility))
                
                # 下午交易（基于上午收盘）
                pm_open = am_close
                pm_volatility = random.uniform(0.008, 0.025)
                pm_close = pm_open * (1 + random.gauss(0, pm_volatility))
                
                # 计算日内高低点
                day_high = max(am_open, am_close, pm_open, pm_close)
                day_low = min(am_open, am_close, pm_open, pm_close)
                
                # 交易量
                volume = random.randint(1000000, 50000000)
                
                morning_open.append(round(am_open, 2))
                morning_close.append(round(am_close, 2))
                afternoon_open.append(round(pm_open, 2))
                afternoon_close.append(round(pm_close, 2))
                volumes.append(volume)
                highs.append(round(day_high, 2))
                lows.append(round(day_low, 2))
                
                # 更新基础价格
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
        """保存股票数据到JSON文件"""
        serializable_data = {}
        
        for stock, df in data.items():
            serializable_data[stock] = {
                'columns': df.columns.tolist(),
                'data': df.to_dict('records')
            }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 股票数据已保存到 {filename}")


class ReinforcementLearningSystem:
    """强化学习系统 - 让智能体从经验中学习"""
    
    def __init__(self):
        self.q_table = {}  # Q-learning表
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.5  # 提高探索率
        self.exploration_decay = 0.99
        
    def get_state_key(self, state_features: Dict) -> str:
        """将状态特征转换为状态键"""
        return json.dumps(state_features, sort_keys=True)
    
    def choose_action(self, state_features: Dict, available_actions: List[str]) -> str:
        """基于当前状态选择动作"""
        state_key = self.get_state_key(state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in available_actions}
        
        # 探索-利用平衡
        if random.random() < self.exploration_rate:
            return random.choice(available_actions)
        else:
            # 如果没有学习数据，随机选择
            if not self.q_table[state_key]:
                return random.choice(available_actions)
            return max(self.q_table[state_key].items(), key=lambda x: x[1])[0]
    
    def update_q_value(self, state_features: Dict, action: str, reward: float, next_state_features: Dict):
        """更新Q值"""
        state_key = self.get_state_key(state_features)
        next_state_key = self.get_state_key(next_state_features)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        
        if next_state_key not in self.q_table:
            next_state_max_q = 0.0
        else:
            next_state_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        current_q = self.q_table[state_key].get(action, 0.0)
        
        # Q-learning更新公式
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * next_state_max_q - current_q)
        self.q_table[state_key][action] = new_q
        
        # 衰减探索率
        self.exploration_rate *= self.exploration_decay
    
    def save_model(self, filename: str = "rl_model.json"):
        """保存强化学习模型"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.q_table, f, indent=2, ensure_ascii=False)
        print(f"💾 强化学习模型已保存到 {filename}")
    
    def load_model(self, filename: str = "rl_model.json"):
        """加载强化学习模型"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                self.q_table = json.load(f)
            print(f"📖 强化学习模型已加载")
        except FileNotFoundError:
            print(f"⚠️  未找到模型文件，使用新的模型")


class TradingStrategyOptimizer:
    """交易策略优化器 - 动态调整交易策略"""
    
    def __init__(self):
        self.strategy_performance = {}
        self.best_strategies = {}
        self.adaptation_rate = 0.2
        
    def record_strategy_performance(self, strategy_name: str, performance: float):
        """记录策略表现"""
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = []
        
        self.strategy_performance[strategy_name].append(performance)
        
        # 只保留最近20个记录
        if len(self.strategy_performance[strategy_name]) > 20:
            self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-20:]
    
    def get_best_strategy(self, context: str) -> str:
        """根据上下文获取最佳策略"""
        if not self.strategy_performance:
            return "default"
        
        if context in self.best_strategies:
            # 如果该上下文有缓存的最佳策略，直接返回
            return self.best_strategies[context]
        
        # 计算所有策略的平均表现
        strategy_scores = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                # 使用指数加权平均，最近的表现更重要
                weights = np.exp(np.linspace(0, 1, len(performances)))
                weighted_avg = np.average(performances, weights=weights)
                strategy_scores[strategy] = weighted_avg
        
        if not strategy_scores:
            return "default"
        
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        self.best_strategies[context] = best_strategy
        
        return best_strategy
    
    def generate_strategy_variation(self, base_strategy: str, creativity: float = 0.3) -> str:
        """生成策略变体"""
        variations = {
            "保守型": ["降低风险偏好", "增加止损点", "减少仓位规模"],
            "激进型": ["提高风险偏好", "放宽止损", "增加仓位规模"],
            "技术型": ["更多技术指标", "关注成交量", "分析市场结构"],
            "情绪型": ["关注市场情绪", "监测新闻事件", "跟踪社交媒体"]
        }
        
        if random.random() < creativity:
            variation_type = random.choice(list(variations.keys()))
            variation = random.choice(variations[variation_type])
            return f"{base_strategy} + {variation}"
        
        return base_strategy


class MarketPatternRecognizer:
    """市场模式识别器 - 学习识别市场模式"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_history = []
        
    def analyze_price_pattern(self, prices: List[float]) -> Dict[str, Any]:
        """分析价格模式"""
        if len(prices) < 5:
            return {"pattern": "unknown", "confidence": 0.0}
        
        # 计算技术指标
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) if len(returns) > 1 else 0
        momentum = (prices[-1] - prices[0]) / prices[0]
        
        # 识别模式
        patterns = []
        
        # 趋势判断
        if len(prices) >= 3:
            short_trend = (prices[-1] - prices[-3]) / prices[-3]
            if abs(short_trend) > 0.01:  # 降低阈值
                trend_type = "uptrend" if short_trend > 0 else "downtrend"
                patterns.append({"name": f"短期{trend_type}", "confidence": min(0.8, abs(short_trend) * 2)})
        
        # 波动率判断
        if volatility > 0.01:  # 降低阈值
            patterns.append({"name": "高波动", "confidence": min(0.9, volatility * 10)})
        elif volatility < 0.008:
            patterns.append({"name": "低波动", "confidence": min(0.9, (0.01 - volatility) * 100)})
        
        # 动量判断
        if abs(momentum) > 0.03:  # 降低阈值
            momentum_type = "强势上涨" if momentum > 0 else "强势下跌"
            patterns.append({"name": momentum_type, "confidence": min(0.85, abs(momentum) * 3)})
        
        # 返回最强的模式
        if patterns:
            strongest_pattern = max(patterns, key=lambda x: x["confidence"])
            return strongest_pattern
        else:
            return {"pattern": "无明确模式", "confidence": 0.5}
    
    def learn_from_pattern(self, pattern: str, outcome: float):
        """从模式结果中学习"""
        if pattern not in self.patterns:
            self.patterns[pattern] = {"outcomes": [], "success_rate": 0.0}
        
        self.patterns[pattern]["outcomes"].append(outcome)
        
        # 计算成功率
        if len(self.patterns[pattern]["outcomes"]) > 0:
            success_count = sum(1 for o in self.patterns[pattern]["outcomes"] if o > 0)
            self.patterns[pattern]["success_rate"] = success_count / len(self.patterns[pattern]["outcomes"])
        
        # 记录历史
        self.pattern_history.append({
            "pattern": pattern,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_pattern_advice(self, pattern: str) -> str:
        """根据模式获取交易建议"""
        if pattern in self.patterns:
            success_rate = self.patterns[pattern]["success_rate"]
            if success_rate > 0.6:
                return f"该模式历史胜率{success_rate:.1%}，建议积极交易"
            elif success_rate < 0.4:
                return f"该模式历史胜率{success_rate:.1%}，建议谨慎操作"
            else:
                return f"该模式历史胜率{success_rate:.1%}，建议观望或小仓位"
        
        return "新模式，建议观察后再决策"


class AIClient:
    """DeepSeek API客户端"""
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.model = "deepseek-chat"
        self.conversation_history = []  # 保存对话历史
    
    def generate_response(self, system_prompt: str, user_prompt: str, use_history: bool = True) -> str:
        """生成AI响应 - 支持对话历史"""
        try:
            messages = [{"role": "system", "content": system_prompt}]
            
            # 添加历史对话（如果启用）
            if use_history and self.conversation_history:
                # 只保留最近的5轮对话
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
            
            # 保存到对话历史
            self.conversation_history.append({"role": "user", "content": user_prompt})
            self.conversation_history.append({"role": "assistant", "content": response_content})
            
            return response_content
        except Exception as e:
            print(f"❌ API调用失败: {e}")
            return f"模拟响应: {system_prompt.split('你是')[1].split('，')[0]}的默认回答"


class BaseTrader:
    """交易者基类 - 增强迭代学习能力"""
    
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
        
        # 新增迭代学习组件
        self.rl_system = ReinforcementLearningSystem()
        self.strategy_optimizer = TradingStrategyOptimizer()
        self.pattern_recognizer = MarketPatternRecognizer()
        self.learning_progress = 0.0  # 学习进度 0-1
        self.adaptation_speed = random.uniform(0.1, 0.3)  # 适应速度
        
        # 智能体元认知
        self.meta_cognition = {
            "strengths": [],
            "weaknesses": [],
            "lessons_learned": [],
            "adaptive_changes": []
        }
    
    def _generate_personality(self) -> Dict[str, Any]:
        """生成交易者个性特征"""
        return {
            "confidence": random.uniform(0.3, 0.9),
            "risk_tolerance": random.uniform(0.4, 0.8),
            "talkativeness": random.uniform(0.5, 0.9),
            "analytical": random.uniform(0.3, 0.8),
            "learning_capacity": random.uniform(0.5, 0.9),  # 学习能力
            "trade_frequency": random.uniform(0.3, 0.7)  # 交易频率倾向
        }
    
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """计算当前投资组合价值"""
        stock_value = sum(shares * current_prices.get(stock, 0) 
                         for stock, shares in self.portfolio.items())
        return self.cash + stock_value
    
    def calculate_weekly_return(self, current_prices: Dict[str, float]) -> float:
        """计算本周收益率 - 修复版"""
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
        """获取性能摘要 - 修复版"""
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
        """分析交易结果并学习"""
        # 提取交易特征
        state_features = {
            "stock": trade_decision.get("stock", ""),
            "action": trade_decision.get("action", ""),
            "market_condition": "unknown",
            "trader_confidence": self.personality_traits["confidence"]
        }
        
        # 定义可用动作
        available_actions = ["buy_small", "buy_medium", "buy_large", 
                           "sell_small", "sell_medium", "sell_large", "hold"]
        
        # 获取实际动作
        action_type = f"{trade_decision.get('action', 'hold')}_{self._get_size_category(trade_decision.get('shares', 0))}"
        
        # 奖励：正收益为正奖励，负收益为负奖励
        reward = outcome * 100  # 放大奖励
        
        # 更新强化学习模型
        self.rl_system.update_q_value(state_features, action_type, reward, state_features)
        
        # 记录策略表现
        strategy_name = f"{self.name}_{trade_decision.get('action', 'hold')}_{trade_decision.get('stock', '')}"
        self.strategy_optimizer.record_strategy_performance(strategy_name, outcome)
        
        # 更新学习进度
        self.learning_progress = min(1.0, self.learning_progress + abs(outcome) * self.adaptation_speed)
    
    def _get_size_category(self, shares: int) -> str:
        """根据股数判断交易规模"""
        if shares <= 3:
            return "small"
        elif shares <= 8:
            return "medium"
        else:
            return "large"
    
    def generate_market_commentary(self, market_data: Dict) -> str:
        """生成市场评论 - 增强版"""
        system_prompt = f"""你是一个{self.name}，请对当前市场状况发表评论。
展现你的个性特点：{self.personality_traits}
你已经从过去的交易中学到了很多，现在的思考更加深入。
用你积累的经验给出独到的市场分析。"""
        
        user_prompt = f"""请评论当前市场：

市场概况：{market_data.get('summary', '暂无数据')}
你的持仓：{self.portfolio}
你的收益率：{self.get_performance_summary()['total_return']:.2%}
你的学习进度：{self.learning_progress:.1%}

请用200字左右发表你的市场观点，特别展示你从经验中学到的东西："""
        
        commentary = self.ai_client.generate_response(system_prompt, user_prompt)
        return commentary

    def discuss_strategy_with(self, other_trader: 'BaseTrader', topic: str) -> str:
        """与其他交易者讨论策略 - 增强版"""
        system_prompt = f"""你是一个{self.name}，正在与{other_trader.name}讨论{topic}。
你是一个有经验的交易者，能够从其他交易者的经验中学习。
展现你通过迭代学习获得的智慧和洞察力。"""
        
        user_prompt = f"""请与{other_trader.name}讨论{topic}：

你的策略：{self.trading_strategy}
你的学习心得：{self._get_recent_lessons()}
对方类型：{other_trader.name}

请进行深入的策略讨论，特别分享你从错误中学到的教训："""
        
        discussion = self.ai_client.generate_response(system_prompt, user_prompt)
        return discussion

    def _get_recent_lessons(self) -> str:
        """获取最近的学习心得"""
        if self.meta_cognition["lessons_learned"]:
            recent_lessons = self.meta_cognition["lessons_learned"][-3:]
            return "\n".join(recent_lessons)
        return "暂无学习心得"

    def react_to_news(self, news: str) -> str:
        """对市场新闻做出反应 - 增强版"""
        system_prompt = f"""你是一个{self.name}，请对以下市场新闻做出反应。
基于你的交易经验和学习成果，给出理性的反应。
展现你通过迭代学习获得的判断力。"""
        
        user_prompt = f"""新闻：{news}

你的持仓：{self.portfolio}
你的策略：{self.trading_strategy}
你的学习进度：{self.learning_progress:.1%}

请发表你的看法和可能的行动，解释你的推理过程："""
        
        reaction = self.ai_client.generate_response(system_prompt, user_prompt)
        return reaction

    def share_experience(self) -> str:
        """分享交易经验 - 增强版"""
        performance = self.get_performance_summary()
        
        system_prompt = f"""你是一个{self.name}，请分享你本周的交易经验和心得体会。
你是一个不断进化的交易者，能够从每次交易中学习。
详细描述你的成长历程和认知进化。"""
        
        user_prompt = f"""请分享你作为{self.name}的交易经验：

【真实业绩数据】
本周收益率:本周收益率: {(self.weekly_returns[-1] if self.weekly_returns else 0):.2%}
总收益率: {performance['total_return']:.2%}
当前持仓价值: {performance['current_portfolio_value']:.2f}
初始资金: {self.initial_capital:.2f}
当前持仓: {self.portfolio}
学习进度: {self.learning_progress:.1%}

【迭代学习成果】
我的成长历程: {self._describe_growth()}
我改正的错误: {self._describe_mistakes()}
我学到的最重要的教训: {self._describe_lessons()}

请详细分享你的心路历程、学习过程和自我提升："""
        
        experience = self.ai_client.generate_response(system_prompt, user_prompt)
        
        # 记录到记忆
        self.memory.append({
            "type": "experience_share",
            "content": experience,
            "week": len(self.weekly_returns),
            "timestamp": datetime.now().isoformat(),
            "learning_progress": self.learning_progress
        })
        
        # 更新元认知
        self._update_meta_cognition(experience)
        
        return experience
    
    def _describe_growth(self) -> str:
        """描述成长历程"""
        if self.learning_progress > 0.7:
            return "我从一个新手成长为有经验的交易者，学会了控制情绪和风险"
        elif self.learning_progress > 0.4:
            return "我正在快速学习，逐渐理解市场规律"
        else:
            return "我还在探索阶段，积累经验中"
    
    def _describe_mistakes(self) -> str:
        """描述改正的错误"""
        mistakes = ["追涨杀跌", "情绪化交易", "忽视风险管理", "过度自信"]
        if self.learning_progress > 0.5:
            learned_mistakes = random.sample(mistakes, 2)
            return f"我已经改正了：{', '.join(learned_mistakes)}"
        elif self.learning_progress > 0.2:
            return f"我正在改正：{random.choice(mistakes)}"
        else:
            return "我还在犯各种错误，需要更多学习"
    
    def _describe_lessons(self) -> str:
        """描述学到的教训"""
        lessons = [
            "风险管理比收益更重要",
            "情绪是交易的最大敌人",
            "耐心等待最佳时机",
            "分散投资降低风险",
            "从错误中学习才能进步"
        ]
        return random.choice(lessons)
    
    def _update_meta_cognition(self, experience: str):
        """更新元认知"""
        # 从经验中提取关键词作为学习点
        keywords = ["学会", "明白", "理解", "发现", "改进", "提升"]
        for keyword in keywords:
            if keyword in experience:
                lesson = experience[experience.find(keyword):experience.find(keyword)+100]
                self.meta_cognition["lessons_learned"].append(lesson[:50] + "...")
                break
    
    def learn_from_others(self, others_experiences: List[Dict]) -> str:
        """从他人经验中学习 - 增强版"""
        if not others_experiences:
            return "本周没有其他交易者分享经验"
        
        experiences_text = "\n\n".join([
            f"{exp['name']}的经验:\n{exp['experience']}" 
            for exp in others_experiences
        ])
        
        system_prompt = f"""你是一个{self.name}，正在学习其他交易者的经验。
你是一个善于学习的交易者，能够批判性地吸收他人经验。
将他人经验与你的自身经验结合，形成更深刻的理解。"""
        
        user_prompt = f"""请分析以下其他交易者的经验分享，并进行深度反思和整合：

{experiences_text}

你的当前策略: {self.trading_strategy}
你的本周收益率: {self.weekly_returns[-1]:.2% if self.weekly_returns else '0%'}
你的学习进度: {self.learning_progress:.1%}

请详细说明：
1. 你从他人经验中学到了什么新东西？
2. 如何将这些经验整合到你的交易哲学中？
3. 具体如何改进你的交易策略？
4. 你计划做出哪些具体的改变？"""
        
        learning = self.ai_client.generate_response(system_prompt, user_prompt)
        
        # 记录学习
        self.memory.append({
            "type": "learning",
            "content": learning,
            "week": len(self.weekly_returns),
            "timestamp": datetime.now().isoformat(),
            "source": "peer_experience"
        })
        
        # 整合学习到策略
        enhanced_strategy = self._integrate_learning(learning)
        self.trading_strategy += f"\n第{len(self.weekly_returns)}周整合学习: {enhanced_strategy}"
        
        # 更新学习进度
        self.learning_progress = min(1.0, self.learning_progress + 0.05)
        
        return learning
    
    def _integrate_learning(self, learning: str) -> str:
        """整合学习到策略中"""
        # 提取关键学习点
        key_phrases = ["学会", "应该", "需要", "改进", "调整", "改变"]
        for phrase in key_phrases:
            if phrase in learning:
                start_idx = learning.find(phrase)
                end_idx = min(start_idx + 80, len(learning))
                return learning[start_idx:end_idx]
        
        return learning[:100]
    
    def summarize_final_experience(self) -> str:
        """总结最终经验 - 增强版"""
        performance = self.get_performance_summary()
        
        system_prompt = f"""你是一个{self.name}，经过一个月的股票交易，请总结你的最终交易经验和进化历程。
你是一个通过迭代学习不断进化的智能交易者。
详细描述你的认知进化、策略改进和心路历程。"""
        
        user_prompt = f"""请总结你作为{self.name}的月度交易经验：

【最终业绩】
最终收益率: {performance['total_return']:.2%}
每周收益率: {[f'{r:.2%}' for r in self.weekly_returns]}
最终组合价值: {performance['current_portfolio_value']:.2f}
初始资金: {self.initial_capital:.2f}
最终持仓: {self.portfolio}
最终学习进度: {self.learning_progress:.1%}

【迭代学习历程】
策略演进: {self.trading_strategy}
关键学习节点: {self._get_key_learnings()}
认知进化: {self._describe_cognitive_evolution()}

请给出深刻的总结，包括：
1. 你的成长曲线
2. 最重要的认知突破
3. 策略的进化过程
4. 对未来的交易哲学的思考"""
        
        final_summary = self.ai_client.generate_response(system_prompt, user_prompt)
        
        self.memory.append({
            "type": "final_summary",
            "content": final_summary,
            "timestamp": datetime.now().isoformat(),
            "final_learning_progress": self.learning_progress
        })
        
        # 保存学习模型
        self._save_learning_models()
        
        return final_summary
    
    def _get_key_learnings(self) -> str:
        """获取关键学习点"""
        key_learnings = []
        for memory_item in self.memory[-5:]:
            if memory_item["type"] in ["learning", "experience_share"]:
                key_learnings.append(memory_item["content"][:50] + "...")
        
        return "\n".join(key_learnings[:3]) if key_learnings else "无记录"
    
    def _describe_cognitive_evolution(self) -> str:
        """描述认知进化"""
        if self.learning_progress > 0.8:
            return "从盲目交易到理性分析，建立了完整的交易体系"
        elif self.learning_progress > 0.5:
            return "开始理解市场规律，学会情绪管理和风险控制"
        elif self.learning_progress > 0.3:
            return "从错误中学习，逐渐形成自己的交易方法"
        else:
            return "仍在探索和试错阶段"
    
    def _save_learning_models(self):
        """保存学习模型"""
        trader_folder = f"trader_{self.name}"
        os.makedirs(trader_folder, exist_ok=True)
        
        # 保存强化学习模型
        self.rl_system.save_model(f"{trader_folder}/rl_model.json")
        
        # 保存策略优化器
        with open(f"{trader_folder}/strategy_optimizer.json", 'w', encoding='utf-8') as f:
            json.dump({
                "strategy_performance": self.strategy_optimizer.strategy_performance,
                "best_strategies": self.strategy_optimizer.best_strategies
            }, f, indent=2, ensure_ascii=False)
        
        # 保存模式识别器
        with open(f"{trader_folder}/pattern_recognizer.json", 'w', encoding='utf-8') as f:
            json.dump({
                "patterns": self.pattern_recognizer.patterns,
                "pattern_history": self.pattern_recognizer.pattern_history[-50:]
            }, f, indent=2, ensure_ascii=False)
        
        print(f"💾 {self.name}的学习模型已保存")
    
    def load_learning_models(self):
        """加载学习模型"""
        trader_folder = f"trader_{self.name}"
        
        # 加载强化学习模型
        self.rl_system.load_model(f"{trader_folder}/rl_model.json")
        
        # 加载策略优化器
        try:
            with open(f"{trader_folder}/strategy_optimizer.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.strategy_optimizer.strategy_performance = data.get("strategy_performance", {})
                self.strategy_optimizer.best_strategies = data.get("best_strategies", {})
        except FileNotFoundError:
            pass
        
        # 加载模式识别器
        try:
            with open(f"{trader_folder}/pattern_recognizer.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.pattern_recognizer.patterns = data.get("patterns", {})
                self.pattern_recognizer.pattern_history = data.get("pattern_history", [])
        except FileNotFoundError:
            pass
        
        print(f"📖 {self.name}的学习模型已加载")


class EmotionalTrader(BaseTrader):
    """情绪投资者 - 增强迭代学习能力"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "情绪投资者")
        self.emotional_state = "neutral"
        self.risk_tolerance = random.uniform(0.6, 0.9)
        self.trading_strategy = "基于市场情绪和价格波动进行交易，容易受到市场情绪影响"
        self.personality_traits.update({
            "emotional_volatility": random.uniform(0.7, 0.95),
            "herd_mentality": random.uniform(0.6, 0.9),
            "impulsiveness": random.uniform(0.6, 0.9)  # 冲动性
        })
        
        # 情绪学习特性
        self.emotional_learning = {
            "panic_threshold": random.uniform(0.6, 0.9),
            "fomo_sensitivity": random.uniform(0.5, 0.8),
            "emotional_resilience": 0.5,  # 情绪恢复力，会通过学习提高
            "mistake_memory": []  # 记住情绪化错误
        }
    
    def make_trading_decisions(self, stock_data: Dict[str, pd.DataFrame], current_day: int) -> List[Dict]:
        """做出交易决策 - 简化版，提高交易频率"""
        decisions = []
        
        # 基础交易概率，第一天更高
        base_trade_prob = 0.3 if current_day == 0 else 0.2
        
        # 根据个性调整交易概率
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"]
        
        # 根据学习进度调整：初期更多交易以积累经验
        if self.learning_progress < 0.3:
            trade_prob *= 1.5
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # 简单的随机交易决策
            if random.random() < trade_prob:
                # 决定买入还是卖出
                if random.random() < 0.6:  # 60%概率买入
                    if self.cash > current_price * 10:
                        shares = random.randint(1, 5)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                elif stock in self.portfolio:  # 有持仓才卖出
                    if random.random() < 0.4:  # 40%概率卖出
                        shares = min(random.randint(1, 3), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """分析交易结果并学习 - 情绪投资者特别版"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        # 如果是情绪化错误，记录下来
        if outcome < -0.05:
            mistake_record = {
                "stock": trade_decision.get("stock", ""),
                "action": trade_decision.get("action", ""),
                "loss": outcome,
                "timestamp": datetime.now().isoformat()
            }
            self.emotional_learning["mistake_memory"].append(mistake_record)
            
            # 只保留最近10个错误
            if len(self.emotional_learning["mistake_memory"]) > 10:
                self.emotional_learning["mistake_memory"] = self.emotional_learning["mistake_memory"][-10:]
        
        # 提高情绪恢复力
        if outcome > 0:
            self.emotional_learning["emotional_resilience"] = min(
                0.9, self.emotional_learning["emotional_resilience"] + 0.02
            )


class RationalFundManager(BaseTrader):
    """理性基金经理 - 增强迭代学习能力"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "理性基金经理")
        self.analysis_depth = random.uniform(0.7, 0.95)
        self.trading_strategy = "基于基本面分析和技术分析进行理性投资决策"
        self.personality_traits.update({
            "analytical": random.uniform(0.8, 0.95),
            "patience": random.uniform(0.7, 0.9),
            "discipline": random.uniform(0.7, 0.9)  # 纪律性
        })
        
        # 理性学习特性
        self.analytical_models = {
            "trend_model_accuracy": 0.5,
            "pattern_recognition_accuracy": 0.5,
            "risk_model_effectiveness": 0.5,
            "optimization_history": []
        }
    
    def make_trading_decisions(self, stock_data: Dict[str, pd.DataFrame], current_day: int) -> List[Dict]:
        """做出交易决策 - 简化版，提高交易频率"""
        decisions = []
        
        # 基础交易概率
        base_trade_prob = 0.25
        
        # 根据个性调整：理性投资者交易更谨慎
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"] * 0.8
        
        # 学习初期更多探索
        if self.learning_progress < 0.4:
            trade_prob *= 1.3
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # 使用简单的趋势分析
            if current_day >= 5:
                recent_prices = df['afternoon_close'].iloc[current_day-5:current_day+1]
                price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                
                # 趋势交易逻辑
                if price_change > 0.01 and random.random() < trade_prob:  # 上涨趋势
                    if self.cash > current_price * 8:
                        shares = random.randint(2, 6)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                
                elif price_change < -0.01 and stock in self.portfolio and random.random() < trade_prob:  # 下跌趋势
                    shares = min(random.randint(1, 4), self.portfolio[stock])
                    if shares > 0:
                        decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
            
            else:
                # 前几天的随机探索
                if random.random() < trade_prob * 1.5:
                    if self.cash > current_price * 10:
                        shares = random.randint(1, 3)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """分析交易结果并学习 - 理性基金经理特别版"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        # 更新分析模型准确性
        if outcome > 0:
            # 成功交易，提高模型信心
            self.analytical_models["trend_model_accuracy"] = min(
                0.95, self.analytical_models["trend_model_accuracy"] + 0.03
            )
        elif outcome < -0.03:
            # 失败交易，稍微降低信心
            self.analytical_models["trend_model_accuracy"] = max(
                0.3, self.analytical_models["trend_model_accuracy"] - 0.01
            )


class InformedTrader(BaseTrader):
    """信息泄露者 - 增强迭代学习能力"""
    
    def __init__(self, trader_id: int):
        super().__init__(trader_id, "信息泄露者")
        self.insider_info = {}
        self.trading_strategy = "利用信息优势进行交易，提前知道某些股票的走势"
        self.personality_traits.update({
            "secretive": random.uniform(0.7, 0.9),
            "opportunistic": random.uniform(0.8, 0.95),
            "aggressive": random.uniform(0.6, 0.9)  # 激进性
        })
        
        # 信息学习特性
        self.info_network = {
            "info_sources": {},
            "info_reliability": 0.7,  # 信息可靠性
            "timing_accuracy": 0.6,   # 时机把握准确性
            "info_history": []        # 信息使用历史
        }
    
    def set_insider_info(self, stock_data: Dict[str, pd.DataFrame]):
        """设置内幕信息 - 简化版"""
        # 选择1-2只股票设置内幕信息
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
        """做出交易决策 - 简化版，提高交易频率"""
        decisions = []
        
        # 基础交易概率（信息泄露者更活跃）
        base_trade_prob = 0.35
        
        # 根据个性调整
        trade_prob = base_trade_prob * self.personality_traits["trade_frequency"] * 1.2
        
        # 学习初期更多探索
        if self.learning_progress < 0.5:
            trade_prob *= 1.4
        
        for stock, df in stock_data.items():
            if current_day >= len(df):
                continue
            
            current_price = df.iloc[current_day]['afternoon_close']
            
            # 1. 首先检查内幕信息
            if stock in self.insider_info:
                info = self.insider_info[stock]
                
                if current_day < info['expiry_day']:
                    if info['direction'] == 'up' and self.cash > current_price * 10:
                        shares = random.randint(3, 8)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                        continue  # 有内幕信息就交易，不执行其他逻辑
                    
                    elif info['direction'] == 'down' and stock in self.portfolio:
                        shares = min(random.randint(3, 6), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
                        continue
            
            # 2. 没有内幕信息时的常规交易
            if random.random() < trade_prob:
                if random.random() < 0.55:  # 55%概率买入
                    if self.cash > current_price * 12:
                        shares = random.randint(2, 5)
                        decisions.append({"action": "buy", "stock": stock, "shares": shares, "price": current_price})
                elif stock in self.portfolio:  # 有持仓才卖出
                    if random.random() < 0.45:  # 45%概率卖出
                        shares = min(random.randint(2, 4), self.portfolio[stock])
                        if shares > 0:
                            decisions.append({"action": "sell", "stock": stock, "shares": shares, "price": current_price})
        
        return decisions
    
    def analyze_trade_outcome(self, trade_decision: Dict, outcome: float):
        """分析交易结果并学习 - 信息泄露者特别版"""
        super().analyze_trade_outcome(trade_decision, outcome)
        
        stock = trade_decision.get("stock", "")
        
        # 更新信息可靠性
        if stock in self.insider_info:
            info = self.insider_info[stock]
            
            # 检查内幕信息的准确性
            if outcome > 0 and info['direction'] == 'up':
                # 成功，提高信息可靠性
                self.info_network["info_reliability"] = min(
                    0.95, self.info_network["info_reliability"] + 0.05
                )
            elif outcome < -0.02 and info['direction'] == 'up':
                # 失败，稍微降低可靠性
                self.info_network["info_reliability"] = max(
                    0.3, self.info_network["info_reliability"] - 0.02
                )
            
            # 记录信息使用历史
            info_record = {
                "stock": stock,
                "info_direction": info['direction'],
                "actual_outcome": outcome,
                "expected_strength": info['strength'],
                "reliability_before": self.info_network["info_reliability"],
                "timestamp": datetime.now().isoformat()
            }
            self.info_network["info_history"].append(info_record)
            
            # 只保留最近记录
            if len(self.info_network["info_history"]) > 10:
                self.info_network["info_history"] = self.info_network["info_history"][-10:]


class TradingSimulation:
    """交易模拟系统 - 迭代学习增强版"""
    
    def __init__(self):
        self.traders = []
        self.stock_data = {}
        self.conversation_log = []
        self.performance_history = []
        self.market_news = self._generate_market_news()
        self.simulation_round = 0  # 模拟轮次
        self.cumulative_learning = {}  # 累计学习数据
    
    def _generate_market_news(self) -> List[Dict]:
        """生成市场新闻"""
        return [
            {"day": 5, "news": "美联储宣布维持利率不变，市场预期稳定"},
            {"day": 12, "news": "科技股财报季来临，多家公司业绩超预期"},
            {"day": 18, "news": "国际油价大幅波动，能源板块受影响"},
            {"day": 25, "news": "监管政策收紧，部分行业面临调整"}
        ]
    
    def initialize_simulation(self, load_previous_learning: bool = True):
        """初始化模拟"""
        print("🚀 初始化迭代学习股票交易模拟系统...")
        
        generator = StockDataGenerator()
        self.stock_data = generator.generate_stock_data(30)
        generator.save_stock_data(self.stock_data)
        
        # 创建增强版交易者
        self.traders = [
            EmotionalTrader(1),
            RationalFundManager(2),
            InformedTrader(3)
        ]
        
        # 加载之前的学习（如果存在）
        if load_previous_learning:
            for trader in self.traders:
                trader.load_learning_models()
        
        for trader in self.traders:
            if isinstance(trader, InformedTrader):
                trader.set_insider_info(self.stock_data)
        
        print("✅ 迭代学习模拟系统初始化完成")
        print(f"📊 股票数量: {len(self.stock_data)}")
        print(f"🤖 交易者: {[trader.name for trader in self.traders]}")
        
        # 显示初始学习状态
        for trader in self.traders:
            print(f"   {trader.name}: 学习进度 {trader.learning_progress:.1%}")
    
    def execute_trades(self, decisions: List[Dict], trader: BaseTrader):
        """执行交易"""
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
                    print(f"   ✅ {trader.name} 买入 {shares}股 {stock} @ {price:.2f}")
            
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
                    print(f"   ✅ {trader.name} 卖出 {shares}股 {stock} @ {price:.2f}")
    
    def analyze_trade_outcomes(self, day_trades: Dict[str, List[Dict]], current_day: int):
        """分析交易结果并让智能体学习"""
        if current_day == 0:
            return
        
        # 获取下一天的价格用于计算收益
        if current_day >= len(list(self.stock_data.values())[0]):
            return
        
        next_day_prices = {}
        for stock, df in self.stock_data.items():
            if current_day < len(df) - 1:
                next_day_prices[stock] = df.iloc[current_day + 1]['afternoon_close']
        
        # 分析每个交易者的交易结果
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
                        # 买入的收益是第二天的价格变化
                        profit = (next_price - price) / price
                    elif action == "sell":
                        # 卖出的收益是避免的损失（假设如果不卖会持有到第二天）
                        profit = (price - next_price) / price  # 注意这是避免的损失
                    else:
                        profit = 0
                    
                    # 让智能体从交易结果中学习
                    trader.analyze_trade_outcome(trade, profit)
    
    def run_market_commentary(self, current_day: int):
        """运行市场评论"""
        print(f"\n📢 第{current_day}天市场评论")
        
        market_data = {
            "summary": f"第{current_day}天交易情况",
            "active_stocks": list(self.stock_data.keys())[:3]
        }
        
        # 选择学习进度最高的交易者发表评论
        commentators = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)[:2]
        for trader in commentators:
            if random.random() < trader.personality_traits["talkativeness"]:
                print(f"\n{trader.name} (学习进度: {trader.learning_progress:.1%}) 发表市场评论:")
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
        """运行策略讨论"""
        print(f"\n💬 第{week}周策略深度讨论")
        
        discussion_topics = [
            "从错误中学习的经验",
            "策略进化的关键节点", 
            "如何平衡风险与收益",
            "市场认知的迭代过程"
        ]
        
        topic = random.choice(discussion_topics)
        print(f"讨论主题: {topic}")
        
        # 选择学习进度最高的两个交易者进行讨论
        participants = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)[:2]
        trader1, trader2 = participants
        
        print(f"\n{trader1.name} (学习进度: {trader1.learning_progress:.1%}) 发起讨论:")
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
        
        print(f"\n{trader2.name} (学习进度: {trader2.learning_progress:.1%}) 回应:")
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
        """运行新闻反应"""
        today_news = [news for news in self.market_news if news["day"] == current_day]
        
        if today_news:
            for news_item in today_news:
                print(f"\n📰 市场新闻: {news_item['news']}")
                
                # 所有交易者对新闻做出反应
                for trader in self.traders:
                    print(f"\n{trader.name} (学习进度: {trader.learning_progress:.1%}) 对新闻的反应:")
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
        """运行每周讨论"""
        print(f"\n🗣️ 第{week}周交易经验分享会")
        
        # 按学习进度排序，让进步最大的先分享
        sorted_traders = sorted(self.traders, key=lambda x: x.learning_progress, reverse=True)
        
        experiences = []
        for trader in sorted_traders:
            print(f"\n{trader.name} (学习进度: {trader.learning_progress:.1%}) 正在分享经验...")
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
        
        # 交易者互相学习（特别关注学习进度低的向高的学习）
        print(f"\n🎓 第{week}周互相学习环节")
        
        # 按学习进度分组
        high_learners = [t for t in self.traders if t.learning_progress > 0.5]
        low_learners = [t for t in self.traders if t.learning_progress <= 0.5]
        
        for learner in low_learners:
            # 让低学习进度者向高学习进度者学习
            if high_learners:
                teacher_experiences = [exp for exp in experiences if exp["name"] in [h.name for h in high_learners]]
                if teacher_experiences:
                    print(f"{learner.name} (学习进度: {learner.learning_progress:.1%}) 正在向高手学习...")
                    
                    learning = learner.learn_from_others(teacher_experiences)
                    self.conversation_log.append({
                        "week": week,
                        "speaker": learner.name,
                        "learning_progress": learner.learning_progress,
                        "type": "learning",
                        "content": learning,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    print(f"{learner.name}的学习心得: {learning[:100]}...")
                    time.sleep(1)
    
    def run_simulation(self, rounds: int = 1):
        """运行完整模拟（支持多轮）"""
        for round_num in range(rounds):
            self.simulation_round = round_num + 1
            print(f"\n🎯 开始第{self.simulation_round}轮股票交易模拟...")
            
            if round_num > 0:
                # 新一轮模拟，保持学习状态但重置部分数据
                print("🔄 开始新一轮模拟，保留学习成果...")
                for trader in self.traders:
                    # 重置现金和持仓，但保持学习模型
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
                print(f"📅 第{week+1}周交易开始 (第{self.simulation_round}轮)")
                print(f"{'='*60}")
                
                # 每周交易
                for day_in_week in range(days_per_week):
                    current_day = week * days_per_week + day_in_week
                    
                    print(f"\n--- 第{current_day+1}天 ---")
                    
                    # 记录当天的交易
                    day_trades = {}
                    
                    # 市场评论（每隔几天一次）
                    if current_day % 3 == 0:
                        self.run_market_commentary(current_day + 1)
                    
                    # 新闻反应
                    self.run_news_reaction(current_day + 1)
                    
                    # 执行交易
                    trade_count = 0
                    for trader in self.traders:
                        decisions = trader.make_trading_decisions(self.stock_data, current_day)
                        day_trades[trader.name] = decisions
                        
                        if decisions:
                            trade_count += len(decisions)
                            self.execute_trades(decisions, trader)
                    
                    print(f"🤝 今日完成 {trade_count} 笔交易")
                    
                    # 分析交易结果并学习
                    self.analyze_trade_outcomes(day_trades, current_day)
                
                # 计算本周收益率
                current_prices = self._get_week_end_prices(week, days_per_week)
                
                print(f"\n💰 第{week+1}周收益率:")
                for trader in self.traders:
                    weekly_return = trader.calculate_weekly_return(current_prices)
                    performance = trader.get_performance_summary()
                    print(f"   {trader.name}: 周收益 {weekly_return:+.2%}, 总收益 {performance['total_return']:+.2%}, 学习进度 {trader.learning_progress:.1%}")
                
                # 策略讨论（每周一次）
                self.run_strategy_discussion(week + 1)
                
                # 每周讨论和学习
                self.run_weekly_discussion(week + 1)
                
                # 记录性能
                self.performance_history.append({
                    "round": self.simulation_round,
                    "week": week + 1,
                    "returns": {trader.name: trader.weekly_returns[-1] for trader in self.traders},
                    "learning_progress": {trader.name: trader.learning_progress for trader in self.traders}
                })
            
            # 最终总结
            self.run_final_summary()
    
    def _get_week_end_prices(self, week: int, days_per_week: int) -> Dict[str, float]:
        """获取周末价格"""
        current_prices = {}
        current_day = (week + 1) * days_per_week - 1
        
        for stock, df in self.stock_data.items():
            if current_day < len(df):
                current_prices[stock] = df.iloc[current_day]['afternoon_close']
        
        return current_prices
    
    def run_final_summary(self):
        """运行最终总结"""
        print("\n🎊 月度交易模拟结束!")
        print("\n📈 最终业绩报告:")
        
        final_returns = {}
        learning_progresses = {}
        final_summaries = []
        
        for trader in self.traders:
            performance = trader.get_performance_summary()
            final_return = performance['total_return']
            final_returns[trader.name] = final_return
            learning_progresses[trader.name] = trader.learning_progress
            
            print(f"\n{trader.name}:")
            print(f"  总收益率: {final_return:.2%}")
            print(f"  学习进度: {trader.learning_progress:.1%}")
            print(f"  最终现金: {trader.cash:.2f}")
            print(f"  最终持仓: {trader.portfolio}")
            print(f"  组合价值: {performance['current_portfolio_value']:.2f}")
            
            print(f"{trader.name} 正在总结最终经验...")
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
            
            print(f"{trader.name}的最终总结: {final_summary[:100]}...")
            time.sleep(1)
        
        # 最佳交易者（综合考虑收益和学习）
        combined_scores = {}
        for name in final_returns.keys():
            # 收益权重0.6，学习进度权重0.4
            return_score = (final_returns[name] + 1) / 2  # 标准化到0-1
            learning_score = learning_progresses[name]
            combined_score = return_score * 0.6 + learning_score * 0.4
            combined_scores[name] = combined_score
        
        best_trader = max(combined_scores.items(), key=lambda x: x[1])
        print(f"\n🏆 本月最佳交易者: {best_trader[0]} (综合得分: {best_trader[1]:.2f})")
        
        # 学习进步奖
        learning_improvement = {name: learning_progresses[name] for name in learning_progresses}
        most_improved = max(learning_improvement.items(), key=lambda x: x[1])
        print(f"📚 学习进步最大: {most_improved[0]} (学习进度: {most_improved[1]:.1%})")
        
        # 保存学习总结
        self.save_learning_summary(final_summaries)
        
        # 保存结果
        self.save_results(final_summaries)
    
    def save_learning_summary(self, final_summaries: List[Dict]):
        """保存学习总结"""
        # 更新累计学习数据
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
        
        # 保存到文件
        with open("cumulative_learning.json", "w", encoding="utf-8") as f:
            json.dump(self.cumulative_learning, f, indent=2, ensure_ascii=False)
    
    def save_results(self, final_summaries: List[Dict]):
        """保存结果到文件"""
        # 保存对话日志
        with open(f"trading_conversations_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(self.conversation_log, f, indent=2, ensure_ascii=False)
        
        # 保存智能体记忆
        memory_data = {}
        for trader in self.traders:
            performance = trader.get_performance_summary()
            memory_data[trader.name] = {
                "final_return": performance['total_return'],
                "learning_progress": trader.learning_progress,
                "current_portfolio_value": performance['current_portfolio_value'],
                "initial_capital": trader.initial_capital,
                "memory": trader.memory[-20:],  # 只保存最近20条记忆
                "trading_strategy": trader.trading_strategy,
                "final_portfolio": trader.portfolio,
                "final_cash": trader.cash,
                "weekly_returns": trader.weekly_returns,
                "personality_traits": trader.personality_traits,
                "meta_cognition": trader.meta_cognition
            }
        
        with open(f"trading_experience_memory_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
        
        # 保存性能历史
        with open(f"trading_performance_round_{self.simulation_round}.json", "w", encoding="utf-8") as f:
            json.dump(self.performance_history, f, indent=2, ensure_ascii=False)
        
        print("✅ 所有结果已保存到JSON文件!")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='迭代学习智能体股票交易模拟系统')
    parser.add_argument('--days', type=int, default=30, help='模拟天数')
    parser.add_argument('--weeks', type=int, default=4, help='模拟周数')
    parser.add_argument('--rounds', type=int, default=1, help='模拟轮次')
    parser.add_argument('--fast', action='store_true', help='快速模式（减少对话）')
    parser.add_argument('--reset-learning', action='store_true', help='重置学习模型')
    
    args = parser.parse_args()
    
    if not os.getenv("DEEPSEEK_API_KEY"):
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        return
    
    print("🧠 启动迭代学习股票交易模拟系统")
    print("=" * 50)
    print("特色功能：")
    print("1. 强化学习 - 智能体从每次交易中学习")
    print("2. 策略优化 - 动态调整交易策略")
    print("3. 模式识别 - 学习识别市场模式")
    print("4. 元认知 - 智能体了解自己的优缺点")
    print("5. 多轮迭代 - 智能体会越来越聪明")
    print("=" * 50)
    
    simulation = TradingSimulation()
    simulation.initialize_simulation(load_previous_learning=not args.reset_learning)
    
    if args.fast:
        print("⚡ 快速模式：简化对话流程")
        # 可以在这里添加简化逻辑
    
    simulation.run_simulation(rounds=args.rounds)
    
    print("\n🎯 模拟完成!")
    print("智能体的学习模型已保存，下次运行时会继续学习")


if __name__ == "__main__":
    main()