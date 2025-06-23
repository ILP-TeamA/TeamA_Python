import azure.functions as func
import logging
import pandas as pd
import numpy as np
import pickle
import os
import requests
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

class EnhancedBeerPredictor:
    def __init__(self, model_dir='trained_models_hyperopt', openweather_api_key=None,
                 azure_db_config=None):
        """
        增强版啤酒销量预测器 - 支持超参数优化模型
        
        Parameters:
        -----------
        model_dir : str
            训练模型存储目录（默认使用超参数优化模型）
        openweather_api_key : str
            OpenWeather API密钥
        azure_db_config : dict
            Azure数据库配置
        """
        # 营业日设置：周一到周六 (0=周一, 1=周二, ..., 5=周六)，周日(6)休息
        self.business_days = [0, 1, 2, 3, 4, 5]
        self.business_day_names = {
            0: '周一', 1: '周二', 2: '周三', 3: '周四', 
            4: '周五', 5: '周六', 6: '周日'
        }
        
        # 如果没有提供配置，使用默认配置（可选）
        if azure_db_config is None:
            azure_db_config = {
                'host': 'teama-db.postgres.database.azure.com',
                'port': '5432',
                'database': 'TeamA_Database',
                'user': 'teamadb',
                'password': 'pass1234!',
                'sslmode': 'require'
            }

        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.beer_types = []
        self.stats = {}
        self.best_params = {}
        self.model_metadata = {}
        self.is_loaded = False
        self.model_dir = model_dir
        self.openweather_api_key = openweather_api_key
        self.azure_db_config = azure_db_config
        self.db_engine = None
        self.historical_data = None
        self._growth_rates = None
        self._customer_growth_rate = None
        
        # OpenWeather API配置
        self.weather_base_url = "http://api.openweathermap.org/data/2.5"
        self.weather_url = f"{self.weather_base_url}/weather"
        self.forecast_url = f"{self.weather_base_url}/forecast"
        
        # 初始化系统
        print("🏗️ 初始化增强版啤酒预测系统 - 支持超参数优化模型...")
        self._init_database_connection()
        self._load_historical_data_from_db()
        self._load_trained_models()
        
        # 计算增长率
        if self.historical_data is not None:
            print("📈 计算增长率...")
            self._growth_rates = self._calculate_growth_rates()
            self._customer_growth_rate = self._calculate_customer_growth_rate()
        
        if self.is_loaded:
            print("✅ 增强版预测系统初始化完成！")
        else:
            raise Exception("❌ 预测系统初始化失败！")
    
    def _init_database_connection(self):
        """初始化数据库连接（可选）"""
        if not self.azure_db_config:
            print("⚠️ 未提供数据库配置，跳过数据库连接")
            return
        
        try:
            connection_string = (
                f"postgresql://{self.azure_db_config['user']}:"
                f"{self.azure_db_config['password']}@"
                f"{self.azure_db_config['host']}:"
                f"{self.azure_db_config['port']}/"
                f"{self.azure_db_config['database']}"
                f"?sslmode={self.azure_db_config.get('sslmode', 'require')}"
            )
            
            self.db_engine = create_engine(connection_string)
            
            with self.db_engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                print("✅ Azure数据库连接成功")
                
        except Exception as e:
            print(f"⚠️ 数据库连接失败: {e}，将使用本地CSV数据或默认值")
            self.db_engine = None
    
    def _load_historical_data_from_db(self):
        """从Azure数据库或本地CSV加载历史数据"""
        if self.db_engine:
            try:
                print("📊 从数据库加载历史数据...")
                
                query = """
                SELECT 
                    dbs.date,
                    dbs.day_of_week as 曜日,
                    dbs.customer_count as 来客数,
                    dbs.total_cups as 総杯数,
                    dbs.total_revenue as 売上合計,
                    p.name as product_name,
                    dbd.quantity,
                    dbd.revenue as product_revenue
                FROM daily_beer_summary dbs
                LEFT JOIN daily_beer_sales dbd ON dbs.sales_id = dbd.sales_id
                LEFT JOIN products p ON dbd.product_id = p.id
                WHERE dbs.date IS NOT NULL
                ORDER BY dbs.date, p.name
                """
                
                raw_data = pd.read_sql(query, self.db_engine)
                
                if len(raw_data) == 0:
                    raise Exception("数据库中没有找到销售数据")
                
                print(f"📊 获取 {len(raw_data)} 条原始记录")
                
                # 数据透视
                pivot_data = raw_data.pivot_table(
                    index=['date', '曜日', '来客数', '総杯数', '売上合計'],
                    columns='product_name',
                    values='quantity',
                    fill_value=0,
                    aggfunc='sum'
                ).reset_index()
                
                # 重命名列
                beer_columns = {}
                for col in pivot_data.columns:
                    if col not in ['date', '曜日', '来客数', '総杯数', '売上合計']:
                        beer_columns[col] = f"{col}(本)"
                
                pivot_data = pivot_data.rename(columns=beer_columns)
                pivot_data['date'] = pd.to_datetime(pivot_data['date'])
                
                # 过滤营业日（周一到周六）
                pivot_data = pivot_data[
                    pivot_data['date'].dt.weekday.isin(self.business_days)
                ].sort_values('date').reset_index(drop=True)
                
                self.historical_data = pivot_data
                self.beer_types = [col.replace('(本)', '') for col in pivot_data.columns 
                                 if col.endswith('(本)')]
                
                print(f"✅ 历史数据加载完成: {len(self.historical_data)}条记录")
                print(f"🍺 啤酒类型: {self.beer_types}")
                return
                    
            except Exception as e:
                print(f"⚠️ 数据库历史数据加载失败: {e}")
        
        # 尝试从本地CSV加载
        try:
            print("📊 尝试从本地CSV加载历史数据...")
            sales_df = pd.read_csv('sales.csv', encoding='utf-8')
            
            sales_df['date'] = pd.to_datetime(sales_df['日付'])
            self.historical_data = sales_df
            self.beer_types = ['ペールエール', 'ラガー', 'IPA', 'ホワイトビール', '黒ビール', 'フルーツビール']
            
            print(f"✅ 本地CSV数据加载完成: {len(self.historical_data)}条记录")
            
        except Exception as e:
            print(f"⚠️ 本地CSV数据加载失败: {e}")
            print("🔄 将使用默认配置")
            self.historical_data = None
            self.beer_types = ['ペールエール', 'ラガー', 'IPA', 'ホワイトビール', '黒ビール', 'フルーツビール']
    
    def _load_trained_models(self):
        """加载增强版训练好的模型（支持超参数优化）"""
        print(f"🤖 加载增强版模型: {self.model_dir}")
        
        # 尝试多个可能的模型目录
        possible_dirs = [
            self.model_dir,
            'trained_models_hyperopt',
            'trained_models_fixed',
            'trained_models'
        ]
        
        model_dir_found = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                model_dir_found = dir_path
                break
        
        if not model_dir_found:
            print(f"⚠️ 所有模型目录都不存在，使用默认模型")
            self._use_enhanced_default_models()
            return
        
        self.model_dir = model_dir_found
        print(f"📁 使用模型目录: {self.model_dir}")
        
        # 加载元数据
        metadata_file = os.path.join(self.model_dir, 'metadata.pkl')
        if not os.path.exists(metadata_file):
            print(f"⚠️ 元数据文件不存在: {metadata_file}，使用默认模型")
            self._use_enhanced_default_models()
            return
        
        try:
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata.get('feature_names', [])
            loaded_beer_types = metadata.get('beer_types', [])
            self.stats = metadata.get('stats', {})
            self.best_params = metadata.get('best_params', {})
            self.model_metadata = metadata
            
            print(f"✅ 元数据加载成功 ({len(self.feature_names)}个特征)")
            print(f"🔧 模型版本: {metadata.get('model_version', 'unknown')}")
            print(f"🔍 搜索方法: {metadata.get('search_method', 'unknown')}")
            
            # 如果模型中有啤酒类型，使用模型的，否则使用历史数据的
            if loaded_beer_types:
                self.beer_types = loaded_beer_types
            
            print(f"🍺 模型啤酒类型: {self.beer_types}")
            
        except Exception as e:
            print(f"❌ 元数据加载失败: {e}，使用默认模型")
            self._use_enhanced_default_models()
            return
        
        # 加载标准化器（如果存在）
        scaler_file = os.path.join(self.model_dir, 'scalers.pkl')
        if os.path.exists(scaler_file):
            try:
                with open(scaler_file, 'rb') as f:
                    self.scalers = pickle.load(f)
                print(f"✅ 标准化器加载成功: {len(self.scalers)}个")
            except Exception as e:
                print(f"⚠️ 标准化器加载失败: {e}")
                self.scalers = {}
        
        # 加载模型文件
        models_loaded = 0
        missing_models = []
        
        for beer in self.beer_types:
            model_file = os.path.join(self.model_dir, f'{beer}_model.pkl')
            
            if not os.path.exists(model_file):
                missing_models.append(beer)
                continue
            
            try:
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                if not hasattr(model, 'predict'):
                    raise ValueError(f"{beer}模型没有predict方法")
                
                self.models[beer] = model
                
                # 显示模型信息
                model_type = type(model).__name__
                best_param = self.best_params.get(beer, {})
                print(f"  ✅ {beer}: {model_type}")
                if best_param:
                    print(f"      最佳参数: {list(best_param.keys())[:3]}...")  # 显示前3个参数
                
                models_loaded += 1
                
            except Exception as e:
                missing_models.append(beer)
                print(f"  ❌ {beer} 加载失败: {e}")
        
        if missing_models or models_loaded == 0:
            print(f"⚠️ 部分模型加载失败: {missing_models}，使用默认模型")
            self._use_enhanced_default_models()
            return
        
        self.is_loaded = len(self.models) > 0
        print(f"✅ 增强版模型加载完成: {models_loaded}/{len(self.beer_types)} 个")
    
    def _use_enhanced_default_models(self):
        """使用增强版默认模型"""
        print("🔄 使用增强版默认预测模型")
        
        # 确保啤酒类型存在
        if not self.beer_types:
            self.beer_types = ['ペールエール', 'ラガー', 'IPA', 'ホワイトビール', '黒ビール', 'フルーツビール']
        
        # 创建增强版默认模型
        class EnhancedDefaultBeerModel:
            def __init__(self, beer_name):
                self.beer_name = beer_name
                self._use_scaling = False
                
                # 基于啤酒类型的更精确基础销量
                self.type_factors = {
                    'ペールエール': {'base': 0.35, 'weekend_boost': 1.3, 'temp_sensitivity': 0.8},
                    'ラガー': {'base': 0.30, 'weekend_boost': 1.2, 'temp_sensitivity': 1.0},
                    'IPA': {'base': 0.20, 'weekend_boost': 1.4, 'temp_sensitivity': 0.6},
                    'ホワイトビール': {'base': 0.15, 'weekend_boost': 1.1, 'temp_sensitivity': 1.2},
                    '黒ビール': {'base': 0.08, 'weekend_boost': 1.0, 'temp_sensitivity': 0.4},
                    'フルーツビール': {'base': 0.06, 'weekend_boost': 1.5, 'temp_sensitivity': 1.3}
                }
            
            def predict(self, X):
                try:
                    if isinstance(X, pd.DataFrame) and len(X) > 0:
                        row = X.iloc[0]
                        customers = row.get('customers', 25)
                        is_weekend = row.get('is_weekend', 0)
                        temp_norm = row.get('avg_temperature_norm', 0)
                        weather_comfort = row.get('weather_comfort', 1.0)
                    elif isinstance(X, dict):
                        customers = X.get('customers', 25)
                        is_weekend = X.get('is_weekend', 0)
                        temp_norm = X.get('avg_temperature_norm', 0)
                        weather_comfort = X.get('weather_comfort', 1.0)
                    else:
                        customers = 25
                        is_weekend = 0
                        temp_norm = 0
                        weather_comfort = 1.0
                    
                    # 获取啤酒类型参数
                    params = self.type_factors.get(self.beer_name, self.type_factors['ペールエール'])
                    
                    # 基础销量
                    base_sales = customers * params['base']
                    
                    # 周末加成
                    if is_weekend:
                        base_sales *= params['weekend_boost']
                    
                    # 温度调整
                    temp_adjustment = 1 + (temp_norm * params['temp_sensitivity'] * 0.1)
                    base_sales *= temp_adjustment
                    
                    # 天气舒适度调整
                    comfort_adjustment = 0.8 + (weather_comfort * 0.4)
                    base_sales *= comfort_adjustment
                    
                    # 确保最小值
                    predicted_sales = max(1.0, base_sales)
                    
                    return [predicted_sales]
                    
                except Exception as e:
                    # 简单备选方案
                    fallback_multipliers = {
                        'ペールエール': 0.35, 'ラガー': 0.30, 'IPA': 0.20,
                        'ホワイトビール': 0.15, '黒ビール': 0.08, 'フルーツビール': 0.06
                    }
                    customers = 25
                    multiplier = fallback_multipliers.get(self.beer_name, 0.2)
                    return [max(1.0, customers * multiplier)]
        
        # 为每种啤酒创建增强版默认模型
        for beer in self.beer_types:
            self.models[beer] = EnhancedDefaultBeerModel(beer)
        
        # 设置增强版特征名称（匹配新训练代码）
        self.feature_names = [
            'customers', 'customers_squared', 'customers_log',
            'customers_low', 'customers_medium', 'customers_high',
            'day_of_week', 'is_weekend', 'month', 'month_sin', 'month_cos',
            'avg_temperature_norm', 'total_rainfall_norm', 'sunshine_hours_norm',
            'weather_comfort', 'customers_temp_interaction', 
            'customers_weekend_interaction', 'customers_comfort_interaction'
        ]
        
        # 设置默认统计信息
        self.stats = {
            'beer_stats': {
                'ペールエール': 8.5, 'ラガー': 7.2, 'IPA': 4.8,
                'ホワイトビール': 3.6, '黒ビール': 2.0, 'フルーツビール': 1.5
            }
        }
        
        self.is_loaded = True
        print(f"✅ 增强版默认模型初始化完成")
    
    def _calculate_growth_rates(self):
        """计算同比增长率"""
        if self.historical_data is None:
            return {beer: 0.05 for beer in self.beer_types}
        
        growth_rates = {}
        current_year = datetime.now().year
        last_year = current_year - 1
        
        for beer in self.beer_types:
            try:
                beer_col = f'{beer}(本)'
                
                if beer_col not in self.historical_data.columns:
                    growth_rates[beer] = 0.05
                    continue
                
                # 今年最近3个月数据
                recent_cutoff = datetime.now() - timedelta(days=90)
                current_year_data = self.historical_data[
                    (self.historical_data['date'] >= recent_cutoff) &
                    (self.historical_data['date'].dt.year == current_year)
                ][beer_col]
                
                # 去年同期数据
                last_year_start = recent_cutoff.replace(year=last_year)
                last_year_end = datetime.now().replace(year=last_year)
                last_year_data = self.historical_data[
                    (self.historical_data['date'] >= last_year_start) &
                    (self.historical_data['date'] <= last_year_end) &
                    (self.historical_data['date'].dt.year == last_year)
                ][beer_col]
                
                if len(current_year_data) > 0 and len(last_year_data) > 0:
                    current_avg = current_year_data.mean()
                    last_year_avg = last_year_data.mean()
                    
                    if last_year_avg > 0:
                        growth = (current_avg - last_year_avg) / last_year_avg
                        growth_rates[beer] = min(max(growth, -0.3), 0.5)
                    else:
                        growth_rates[beer] = 0.05
                else:
                    growth_rates[beer] = 0.05
                    
            except Exception as e:
                growth_rates[beer] = 0.05
        
        return growth_rates

    def _calculate_customer_growth_rate(self):
        """计算来客数同比增长率"""
        if self.historical_data is None or '来客数' not in self.historical_data.columns:
            return 0.03
        
        current_year = datetime.now().year
        last_year = current_year - 1
        
        try:
            recent_cutoff = datetime.now() - timedelta(days=90)
            current_year_customers = self.historical_data[
                (self.historical_data['date'] >= recent_cutoff) &
                (self.historical_data['date'].dt.year == current_year)
            ]['来客数']
            
            last_year_start = recent_cutoff.replace(year=last_year)
            last_year_end = datetime.now().replace(year=last_year)
            last_year_customers = self.historical_data[
                (self.historical_data['date'] >= last_year_start) &
                (self.historical_data['date'] <= last_year_end) &
                (self.historical_data['date'].dt.year == last_year)
            ]['来客数']
            
            if len(current_year_customers) > 0 and len(last_year_customers) > 0:
                current_avg = current_year_customers.mean()
                last_year_avg = last_year_customers.mean()
                
                if last_year_avg > 0:
                    customer_growth = (current_avg - last_year_avg) / last_year_avg
                    return min(max(customer_growth, -0.2), 0.3)
            
            return 0.03
            
        except Exception as e:
            return 0.03

    def _estimate_customer_count(self, target_date, weather_data):
        """
        改进版来客数预测 - 基于去年同期 + 天气调整
        """
        if self.historical_data is None or '来客数' not in self.historical_data.columns:
            return self._estimate_customer_by_weather_and_date(target_date, weather_data)
        
        pred_date = pd.to_datetime(target_date)
        weekday = pred_date.weekday()
        
        try:
            # 获取去年同期基础数据
            base_customers = self._get_last_year_same_period_customers(pred_date, weekday)
            
            # 天气调整系数
            weather_factor = self._calculate_weather_adjustment_factor(weather_data)
            
            # 季节趋势调整
            seasonal_factor = self._get_seasonal_trend_factor(pred_date)
            
            # 应用增长率
            growth_factor = 1 + (self._customer_growth_rate or 0.03)
            
            # 综合计算
            adjusted_customers = base_customers * weather_factor * seasonal_factor * growth_factor
            
            # 确保结果在合理范围内
            adjusted_customers = max(5, min(100, adjusted_customers))
            
            return int(adjusted_customers)
            
        except Exception as e:
            return self._estimate_customer_by_weather_and_date(target_date, weather_data)
    
    def _estimate_customer_by_weather_and_date(self, target_date, weather_data):
        """基于天气和日期的智能来客数估算（无历史数据时使用）"""
        pred_date = pd.to_datetime(target_date)
        weekday = pred_date.weekday()
        
        # 基础来客数（按星期几）
        base_customers_by_weekday = {
            0: 22,  # 周一
            1: 25,  # 周二
            2: 23,  # 周三
            3: 26,  # 周四
            4: 28,  # 周五
            5: 30,  # 周六
        }
        
        base_customers = base_customers_by_weekday.get(weekday, 25)
        
        # 天气调整
        weather_factor = self._calculate_weather_adjustment_factor(weather_data)
        
        # 季节调整
        seasonal_factor = self._get_seasonal_trend_factor(pred_date)
        
        final_customers = int(base_customers * weather_factor * seasonal_factor)
        final_customers = max(10, min(50, final_customers))
        
        return final_customers

    def _get_last_year_same_period_customers(self, pred_date, weekday):
        """获取去年同期的来客数基础值"""
        last_year_date = pred_date.replace(year=pred_date.year - 1)
        
        # 方法1: 精确匹配去年同一天（±7天范围内的同星期几）
        date_range_start = last_year_date - timedelta(days=7)
        date_range_end = last_year_date + timedelta(days=7)
        
        same_period_data = self.historical_data[
            (self.historical_data['date'] >= date_range_start) &
            (self.historical_data['date'] <= date_range_end) &
            (self.historical_data['date'].dt.weekday == weekday)
        ]
        
        if len(same_period_data) > 0:
            base_customers = same_period_data['来客数'].mean()
            return base_customers
        
        # 方法2: 去年同月同星期几的平均值
        last_year_month_data = self.historical_data[
            (self.historical_data['date'].dt.year == pred_date.year - 1) &
            (self.historical_data['date'].dt.month == pred_date.month) &
            (self.historical_data['date'].dt.weekday == weekday)
        ]
        
        if len(last_year_month_data) > 0:
            base_customers = last_year_month_data['来客数'].mean()
            return base_customers
        
        # 方法3: 历史同星期几平均值
        all_weekday_data = self.historical_data[
            self.historical_data['date'].dt.weekday == weekday
        ]
        
        if len(all_weekday_data) > 0:
            base_customers = all_weekday_data['来客数'].mean()
            return base_customers
        
        # 最终备选
        return 25.0

    def _calculate_weather_adjustment_factor(self, weather_data):
        """计算天气调整系数"""
        temp = weather_data.get('avg_temperature', weather_data.get('平均気温(℃)', 20))
        rainfall = weather_data.get('total_rainfall', weather_data.get('降水量の合計(mm)', 0))
        sunshine = weather_data.get('sunshine_hours', weather_data.get('日照時間(時間)', 6))
        
        # 1. 温度因子 (0.7 - 1.2)
        if 18 <= temp <= 25:
            temp_factor = 1.1
        elif 15 <= temp <= 30:
            temp_factor = 1.0
        elif 10 <= temp <= 35:
            temp_factor = 0.9
        elif temp > 35 or temp < 5:
            temp_factor = 0.7
        else:
            temp_factor = 0.8
        
        # 2. 降雨因子 (0.6 - 1.0)
        if rainfall == 0:
            rain_factor = 1.0
        elif rainfall <= 2:
            rain_factor = 0.9
        elif rainfall <= 10:
            rain_factor = 0.8
        elif rainfall <= 25:
            rain_factor = 0.7
        else:
            rain_factor = 0.6
        
        # 3. 日照因子 (0.8 - 1.2)
        if sunshine >= 8:
            sun_factor = 1.2
        elif sunshine >= 6:
            sun_factor = 1.0
        elif sunshine >= 3:
            sun_factor = 0.9
        else:
            sun_factor = 0.8
        
        weather_factor = (temp_factor * 0.4 + rain_factor * 0.4 + sun_factor * 0.2)
        return max(0.6, min(1.4, weather_factor))

    def _get_seasonal_trend_factor(self, pred_date):
        """获取季节趋势调整系数"""
        month = pred_date.month
        
        seasonal_factors = {
            1: 0.85, 2: 0.85, 3: 1.05, 4: 1.15, 5: 1.10, 6: 1.10,
            7: 1.15, 8: 1.05, 9: 1.10, 10: 1.05, 11: 1.00, 12: 0.90
        }
        
        return seasonal_factors.get(month, 1.0)
    
    def get_weather_data(self, target_date, city="Tokyo", country_code="JP"):
        """获取天气数据"""
        if not self.openweather_api_key:
            return self._get_default_weather()
        
        try:
            target_dt = pd.to_datetime(target_date)
            today = datetime.now().date()
            target_date_obj = target_dt.date()
            
            if target_date_obj < today:
                return self._get_historical_weather(target_dt, city, country_code)
            elif target_date_obj == today:
                return self._get_current_weather(city, country_code)
            else:
                return self._get_forecast_weather(target_dt, city, country_code)
                
        except Exception as e:
            return self._get_default_weather()
    
    def _get_historical_weather(self, target_dt, city, country_code):
        """获取历史天气数据"""
        target_date_str = target_dt.strftime('%Y-%m-%d')
        
        if self.db_engine:
            try:
                query = """
                SELECT avg_temperature, max_temperature, min_temperature,
                       total_rainfall, sunshine_hours, avg_humidity
                FROM weather_history 
                WHERE date = :target_date LIMIT 1
                """
                
                with self.db_engine.connect() as conn:
                    result = conn.execute(text(query), {"target_date": target_date_str})
                    weather_row = result.fetchone()
                
                if weather_row:
                    return {
                        'avg_temperature': float(weather_row[0]) if weather_row[0] is not None else 20.0,
                        'max_temperature': float(weather_row[1]) if weather_row[1] is not None else 25.0,
                        'min_temperature': float(weather_row[2]) if weather_row[2] is not None else 15.0,
                        'total_rainfall': float(weather_row[3]) if weather_row[3] is not None else 0.0,
                        'sunshine_hours': float(weather_row[4]) if weather_row[4] is not None else 6.0,
                        'avg_humidity': float(weather_row[5]) if weather_row[5] is not None else 60.0,
                        'description': '历史天气'
                    }
            except Exception as e:
                print(f"❌ 数据库历史天气查询失败: {e}")
        
        return self._get_seasonal_default_weather(target_dt)
    
    def _get_seasonal_default_weather(self, target_dt):
        """基于季节的智能默认天气数据"""
        month = target_dt.month
        
        if month in [12, 1, 2]:
            temp_base = 8
            rain_prob = 0.3
        elif month in [3, 4, 5]:
            temp_base = 18
            rain_prob = 0.4
        elif month in [6, 7, 8]:
            temp_base = 26
            rain_prob = 0.5
        else:
            temp_base = 20
            rain_prob = 0.2
        
        temp_variation = np.random.uniform(-3, 3)
        rainfall = np.random.uniform(0, 5) if np.random.random() < rain_prob else 0
        
        return {
            'avg_temperature': temp_base + temp_variation,
            'max_temperature': temp_base + temp_variation + 4,
            'min_temperature': temp_base + temp_variation - 4,
            'total_rainfall': rainfall,
            'sunshine_hours': np.random.uniform(3, 9) if rainfall < 1 else np.random.uniform(1, 4),
            'avg_humidity': np.random.uniform(50, 80),
            'description': '多云' if rainfall < 1 else '小雨'
        }
    
    def _get_default_weather(self):
        """默认天气数据"""
        month = datetime.now().month
        if month in [12, 1, 2]:
            temp_base = 8
        elif month in [3, 4, 5]:
            temp_base = 18
        elif month in [6, 7, 8]:
            temp_base = 26
        else:
            temp_base = 20
        
        return {
            'avg_temperature': temp_base + np.random.uniform(-2, 2),
            'max_temperature': temp_base + 4,
            'min_temperature': temp_base - 4,
            'total_rainfall': np.random.uniform(0, 3),
            'sunshine_hours': np.random.uniform(5, 9),
            'avg_humidity': np.random.uniform(55, 75),
            'description': '晴转多云'
        }
    
    def _get_current_weather(self, city, country_code):
        """获取当前天气"""
        try:
            params = {
                'q': f"{city},{country_code}",
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.weather_url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            return {
                'avg_temperature': weather_data['main']['temp'],
                'max_temperature': weather_data['main']['temp_max'],
                'min_temperature': weather_data['main']['temp_min'],
                'total_rainfall': weather_data.get('rain', {}).get('1h', 0) * 24,
                'sunshine_hours': 8 if weather_data['clouds']['all'] < 30 else 4,
                'avg_humidity': weather_data['main']['humidity'],
                'description': weather_data['weather'][0]['description']
            }
            
        except Exception as e:
            return self._get_default_weather()
    
    def _get_forecast_weather(self, target_dt, city, country_code):
        """获取天气预测"""
        try:
            days_ahead = (target_dt.date() - datetime.now().date()).days
            
            if days_ahead > 5:
                return self._get_seasonal_default_weather(target_dt)
            
            params = {
                'q': f"{city},{country_code}",
                'appid': self.openweather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            target_date_str = target_dt.strftime('%Y-%m-%d')
            
            daily_data = []
            for item in forecast_data['list']:
                dt = datetime.fromtimestamp(item['dt'])
                if dt.strftime('%Y-%m-%d') == target_date_str:
                    daily_data.append(item)
            
            if not daily_data:
                daily_data = [forecast_data['list'][min(days_ahead * 8, len(forecast_data['list'])-1)]]
            
            avg_temp = sum(item['main']['temp'] for item in daily_data) / len(daily_data)
            max_temp = max(item['main']['temp_max'] for item in daily_data)
            min_temp = min(item['main']['temp_min'] for item in daily_data)
            total_rain = sum(item.get('rain', {}).get('3h', 0) for item in daily_data)
            avg_humidity = sum(item['main']['humidity'] for item in daily_data) / len(daily_data)
            avg_clouds = sum(item['clouds']['all'] for item in daily_data) / len(daily_data)
            
            return {
                'avg_temperature': avg_temp,
                'max_temperature': max_temp,
                'min_temperature': min_temp,
                'total_rainfall': total_rain,
                'sunshine_hours': 8 if avg_clouds < 30 else (4 if avg_clouds < 70 else 2),
                'avg_humidity': avg_humidity,
                'description': daily_data[0]['weather'][0]['description']
            }
            
        except Exception as e:
            return self._get_seasonal_default_weather(target_dt)
    
    def _build_enhanced_features(self, target_date, weather_data, customer_count):
        """构建增强特征 - 完全匹配超参数优化训练模型的特征工程"""
        pred_date = pd.to_datetime(target_date)
        weather_data = self._standardize_weather_input(weather_data)
        
        features = {}
        
        # 🔥 第1优先级：强化来客数特征（完全匹配训练代码）
        features['customers'] = customer_count
        features['customers_squared'] = customer_count ** 2
        features['customers_log'] = np.log1p(customer_count)
        features['customers_sqrt'] = np.sqrt(customer_count)
        
        # 来客数分级特征
        features['customers_low'] = 1 if customer_count < 15 else 0
        features['customers_medium'] = 1 if 15 <= customer_count < 25 else 0
        features['customers_high'] = 1 if customer_count >= 25 else 0
        
        # 来客数标准化
        customers_mean = 25.0  # 假设平均值
        customers_std = 8.0    # 假设标准差
        features['customers_normalized'] = (customer_count - customers_mean) / customers_std
        
        # 🔥 第2优先级：核心时间特征
        features['day_of_week'] = pred_date.weekday() + 1
        features['is_weekend'] = 1 if pred_date.weekday() >= 5 else 0
        features['is_friday'] = 1 if pred_date.weekday() == 4 else 0
        features['month'] = pred_date.month
        
        # 季节性编码
        features['month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
        features['day_sin'] = np.sin(2 * np.pi * (pred_date.weekday() + 1) / 7)
        features['day_cos'] = np.cos(2 * np.pi * (pred_date.weekday() + 1) / 7)
        
        # 🔥 第3优先级：关键天气特征
        temp_mean, temp_std = 20.0, 10.0
        rain_mean, rain_std = 3.0, 5.0
        sun_mean, sun_std = 6.0, 3.0
        
        if 'avg_temperature' in weather_data:
            temp = weather_data['avg_temperature']
            features['avg_temperature_norm'] = (temp - temp_mean) / temp_std
            features['avg_temperature_squared'] = temp ** 2
            
            # 温度分级
            features['temp_cold'] = 1 if temp < 10 else 0
            features['temp_cool'] = 1 if 10 <= temp < 20 else 0
            features['temp_warm'] = 1 if 20 <= temp < 30 else 0
            features['temp_hot'] = 1 if temp >= 30 else 0
        
        if 'total_rainfall' in weather_data:
            rainfall = weather_data['total_rainfall']
            features['total_rainfall_norm'] = (rainfall - rain_mean) / rain_std
            features['total_rainfall_squared'] = rainfall ** 2
        
        if 'sunshine_hours' in weather_data:
            sunshine = weather_data['sunshine_hours']
            features['sunshine_hours_norm'] = (sunshine - sun_mean) / sun_std
            features['sunshine_hours_squared'] = sunshine ** 2
        
        # 天气舒适度指数
        if all(f'{f}_norm' in features for f in ['avg_temperature', 'total_rainfall', 'sunshine_hours']):
            features['weather_comfort'] = (
                (1 - abs(features['avg_temperature_norm'])) * 
                (1 - abs(features['total_rainfall_norm'])) *
                (1 + features['sunshine_hours_norm'])
            )
        else:
            features['weather_comfort'] = 1.0
        
        # 🔥 第4优先级：来客数交互特征
        if 'avg_temperature_norm' in features:
            features['customers_temp_interaction'] = customer_count * features['avg_temperature_norm']
        
        features['customers_weekend_interaction'] = customer_count * features['is_weekend']
        features['customers_comfort_interaction'] = customer_count * features['weather_comfort']
        features['customers_month_interaction'] = customer_count * features['month']
        
        # 🔥 第5优先级：滞后特征（设为0，因为没有历史数据）
        for beer in self.beer_types:
            features[f'{beer}_lag1'] = 0
            features[f'{beer}_ma3'] = 0
            features[f'{beer}_ma7'] = 0
            features[f'{beer}_customer_ratio_ma3'] = 0
        
        # 来客数滞后
        features['customers_lag1'] = 0
        features['customers_ma3'] = 0
        features['customers_ma7'] = 0
        
        return features
    
    def _standardize_weather_input(self, weather_data):
        """标准化天气数据"""
        mapping = {
            'avg_temperature': '平均気温(℃)',
            'max_temperature': '最高気温(℃)',
            'min_temperature': '最低気温(℃)',
            'total_rainfall': '降水量の合計(mm)',
            'sunshine_hours': '日照時間(時間)',
            'avg_humidity': '平均湿度(％)'
        }
        
        standardized = {}
        for eng_key, jp_key in mapping.items():
            if eng_key in weather_data:
                standardized[eng_key] = weather_data[eng_key]
                standardized[jp_key] = weather_data[eng_key]
            elif jp_key in weather_data:
                standardized[eng_key] = weather_data[jp_key]
                standardized[jp_key] = weather_data[jp_key]
        
        defaults = {
            'avg_temperature': 22.0, 'max_temperature': 27.0, 'min_temperature': 17.0,
            'total_rainfall': 0.0, 'sunshine_hours': 6.0, 'avg_humidity': 60.0,
            '平均気温(℃)': 22.0, '最高気温(℃)': 27.0, '最低気温(℃)': 17.0,
            '降水量の合計(mm)': 0.0, '日照時間(時間)': 6.0, '平均湿度(％)': 60.0
        }
        
        for key, default_val in defaults.items():
            if key not in standardized:
                standardized[key] = default_val
        
        return standardized

    def predict_date_sales(self, target_date, city="Tokyo", country_code="JP", 
                          customer_count=None, save_json=False, save_dir="predictions"):
        """预测指定日期销量 - 使用增强版模型"""
        
        if not self.is_loaded:
            raise Exception('增强版预测系统未正确初始化')
        
        try:
            target_dt = pd.to_datetime(target_date)
            target_date_str = target_dt.strftime('%Y-%m-%d')
            weekday = target_dt.weekday()
            day_name = self.business_day_names[weekday]
            
            if weekday not in self.business_days:
                return {
                    'success': False,
                    'error': f'{target_date_str}是{day_name}，非营业日',
                    'error_code': 'NON_BUSINESS_DAY'
                }
            
            # 获取天气数据
            weather_data = self.get_weather_data(target_date, city, country_code)
            
            # 来客数预测
            if customer_count is None:
                customer_count = self._estimate_customer_count(target_date, weather_data)
            
            # 构建增强特征
            features = self._build_enhanced_features(target_date, weather_data, customer_count)
            
            # 转换为DataFrame
            feature_df = pd.DataFrame([features])
            
            # 确保所有模型需要的特征都存在
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # 执行增强版预测
            daily_predictions = {}
            raw_predictions = {}
            model_details = {}
            
            for beer, model in self.models.items():
                try:
                    # 准备特征数据
                    if self.feature_names:
                        X = feature_df[self.feature_names].fillna(0)
                    else:
                        X = feature_df
                    
                    # 检查是否需要标准化
                    scaler_key = f'{beer}_scaler'
                    if hasattr(model, '_use_scaling') and model._use_scaling:
                        if hasattr(model, '_scaler'):
                            X_scaled = model._scaler.transform(X)
                            pred_sales = model.predict(X_scaled)[0]
                        elif scaler_key in self.scalers:
                            X_scaled = self.scalers[scaler_key].transform(X)
                            pred_sales = model.predict(X_scaled)[0]
                        else:
                            pred_sales = model.predict(X)[0]
                    else:
                        pred_sales = model.predict(X)[0]
                    
                    # 记录模型详情
                    model_details[beer] = {
                        'model_type': type(model).__name__,
                        'uses_scaling': getattr(model, '_use_scaling', False),
                        'best_params': self.best_params.get(beer, {})
                    }
                    
                    # 应用增长率调整
                    if self._growth_rates and beer in self._growth_rates:
                        growth_factor = 1 + self._growth_rates[beer]
                        pred_sales *= growth_factor
                    
                    # 保存原始预测值
                    raw_pred = max(0.1, pred_sales)
                    raw_predictions[beer] = raw_pred
                    
                    # 应用1.2倍系数作为进货建议量
                    purchase_recommendation = raw_pred * 1.2
                    daily_predictions[beer] = purchase_recommendation
                    
                except Exception as e:
                    logging.error(f"模型预测失败 - {beer}: {e}")
                    # 使用统计备选方案
                    avg_sales = self.stats.get('beer_stats', {}).get(beer, 5.0)
                    raw_pred = max(1.0, avg_sales)
                    raw_predictions[beer] = raw_pred
                    daily_predictions[beer] = raw_pred * 1.2
                    
                    model_details[beer] = {
                        'model_type': 'Fallback',
                        'error': str(e)
                    }
            
            # 构建增强版结果
            result = {
                'success': True,
                'prediction_date': target_date_str,
                'day_of_week': day_name,
                'weather_temperature': f"{weather_data.get('avg_temperature', 22):.1f}°C",
                'weather_description': weather_data.get('description', '晴转多云'),
                'estimated_customers': customer_count,
                'beer_purchase_recommendations': {beer: round(amount, 1) for beer, amount in daily_predictions.items()},
                'beer_sales_predictions': {beer: round(amount, 1) for beer, amount in raw_predictions.items()},
                'total_purchase_recommendation': round(sum(daily_predictions.values()), 1),
                'total_predicted_sales': round(sum(raw_predictions.values()), 1),
                'purchase_multiplier': 1.2,
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_info': {
                    'model_directory': self.model_dir,
                    'model_version': self.model_metadata.get('model_version', 'unknown'),
                    'search_method': self.model_metadata.get('search_method', 'unknown'),
                    'feature_count': len(self.feature_names),
                    'model_details': model_details
                }
            }
            
            return result
            
        except Exception as e:
            logging.error(f"增强版预测失败: {e}")
            return {
                'success': False,
                'error': f'增强版预测失败: {str(e)}',
                'error_code': 'ENHANCED_PREDICTION_FAILED'
            }


# 全局预测器实例
predictor = None

def get_predictor():
    """获取增强版预测器实例（单例模式）"""
    global predictor
    if predictor is None:
        try:
            OPENWEATHER_API_KEY = "28425f53a2d3a84301b3bcf4e7b7b203"
            predictor = EnhancedBeerPredictor(
                model_dir='trained_models_hyperopt',  # 优先使用超参数优化模型
                openweather_api_key=OPENWEATHER_API_KEY
            )
        except Exception as e:
            logging.error(f"增强版预测器初始化失败: {e}")
            try:
                # 回退到固定版模型
                predictor = EnhancedBeerPredictor(
                    model_dir='trained_models_fixed',
                    openweather_api_key=OPENWEATHER_API_KEY
                )
            except Exception as e2:
                # 最终回退到默认模型
                predictor = EnhancedBeerPredictor(
                    model_dir='nonexistent',  # 触发使用默认模型
                    openweather_api_key=OPENWEATHER_API_KEY
                )
    return predictor

def get_monday_order_predictions(base_date):
    """获取周一的订货建议（周二、周三、周四三天的总和）"""
    predictor = get_predictor()
    base_dt = pd.to_datetime(base_date)
    
    # 周一预测：周二(+1)、周三(+2)、周四(+3)
    prediction_dates = []
    for days_ahead in [1, 2, 3]:  # 周二、周三、周四
        prediction_dates.append(base_dt + timedelta(days=days_ahead))
    
    order_predictions = {}
    total_by_beer = {}
    
    for beer in predictor.beer_types:
        total_by_beer[beer] = 0
    
    for date in prediction_dates:
        try:
            result = predictor.predict_date_sales(
                target_date=date.strftime('%Y-%m-%d'),
                save_json=False
            )
            
            if result and result.get('success'):
                date_str = result['prediction_date']
                day_name = result['day_of_week']
                
                order_predictions[date_str] = {
                    'date': date_str,
                    'day_of_week': day_name,
                    'weather_temperature': result['weather_temperature'],
                    'weather_description': result.get('weather_description', ''),
                    'estimated_customers': result['estimated_customers'],
                    'beer_orders': result['beer_purchase_recommendations'],
                    'total_order': result['total_purchase_recommendation']
                }
                
                # 累计各啤酒的总订货量
                for beer, amount in result['beer_purchase_recommendations'].items():
                    total_by_beer[beer] += amount
                    
        except Exception as e:
            logging.error(f"预测日期 {date.strftime('%Y-%m-%d')} 失败: {e}")
    
    return {
        'type': 'monday_order_recommendations',
        'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_date': base_date,
        'order_period': '周二至周四 (3天)',
        'model_info': predictor.model_metadata.get('model_version', 'enhanced_predictor'),
        'daily_predictions': order_predictions,
        'beer_purchase_recommendations': {beer: round(amount, 1) for beer, amount in total_by_beer.items()},
        'total_order_amount': round(sum(total_by_beer.values()), 1)
    }

def get_thursday_order_predictions(base_date):
    """获取周四的订货建议（周五、周六、下周一三天的总和）"""
    predictor = get_predictor()
    base_dt = pd.to_datetime(base_date)
    
    # 周四预测：周五(+1)、周六(+2)、下周一(+4，跳过周日)
    prediction_dates = []
    for days_ahead in [1, 2, 4]:  # 周五、周六、下周一
        prediction_dates.append(base_dt + timedelta(days=days_ahead))
    
    order_predictions = {}
    total_by_beer = {}
    
    for beer in predictor.beer_types:
        total_by_beer[beer] = 0
    
    for date in prediction_dates:
        try:
            result = predictor.predict_date_sales(
                target_date=date.strftime('%Y-%m-%d'),
                save_json=False
            )
            
            if result and result.get('success'):
                date_str = result['prediction_date']
                day_name = result['day_of_week']
                
                order_predictions[date_str] = {
                    'date': date_str,
                    'day_of_week': day_name,
                    'weather_temperature': result['weather_temperature'],
                    'weather_description': result.get('weather_description', ''),
                    'estimated_customers': result['estimated_customers'],
                    'beer_orders': result['beer_purchase_recommendations'],
                    'total_order': result['total_purchase_recommendation']
                }
                
                # 累计各啤酒的总订货量
                for beer, amount in result['beer_purchase_recommendations'].items():
                    total_by_beer[beer] += amount
                    
        except Exception as e:
            logging.error(f"预测日期 {date.strftime('%Y-%m-%d')} 失败: {e}")
    
    return {
        'type': 'thursday_order_recommendations',
        'generated_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'base_date': base_date,
        'order_period': '周五至周六及下周一 (3天)',
        'model_info': predictor.model_metadata.get('model_version', 'enhanced_predictor'),
        'daily_predictions': order_predictions,
        'beer_purchase_recommendations': {beer: round(amount, 1) for beer, amount in total_by_beer.items()},
        'total_order_amount': round(sum(total_by_beer.values()), 1)
    }

@app.function_name(name="predict_beer")
@app.route(route="predictor", methods=["GET", "POST"])
def predictor_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """增强版啤酒预测API端点 - 新业务逻辑：仅周一和周四可下单"""
    logging.info('增强版啤酒预测API被调用')
    
    try:
        # 解析请求参数
        target_date = None
        
        if req.method == "POST":
            # 处理POST请求 - 从JSON获取参数
            try:
                req_body = req.get_json()
                if req_body:
                    target_date = req_body.get('target_date')
                    logging.info(f'从POST JSON获取目标日期: {target_date}')
                else:
                    logging.warning('POST请求体为空')
            except Exception as e:
                logging.error(f'解析POST JSON失败: {e}')
        
        if req.method == "GET" or not target_date:
            # 处理GET请求或POST请求无有效日期 - 从查询参数获取
            target_date = req.params.get('target_date')
            logging.info(f'从GET参数获取目标日期: {target_date}')
        
        # 如果没有指定日期，使用当前日期
        if not target_date:
            target_date = datetime.now().strftime('%Y-%m-%d')
            logging.info(f'使用当前日期: {target_date}')
        
        # 验证日期格式
        try:
            target_dt = pd.to_datetime(target_date)
            target_date_str = target_dt.strftime('%Y-%m-%d')
            weekday = target_dt.weekday()  # 0=周一, 1=周二, ..., 6=周日
            day_name = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}[weekday]
            
            logging.info(f'处理日期: {target_date_str} ({day_name})')
        except Exception as e:
            return func.HttpResponse(
                json.dumps({
                    'error': f'日期格式错误: {target_date}，请使用YYYY-MM-DD格式',
                    'error_code': 'INVALID_DATE_FORMAT',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'beer_purchase_recommendations': {}
                }, ensure_ascii=False, indent=2),
                status_code=400,
                mimetype="application/json"
            )
        
        # 🔥 新业务逻辑：仅周一和周四可下单
        if weekday == 0:  # 周一 - 返回周二、周三、周四的订货建议
            response_data = get_monday_order_predictions(target_date_str)
            logging.info('处理周一订货建议 (周二-周四)')
            
        elif weekday == 3:  # 周四 - 返回周五、周六、下周一的订货建议
            response_data = get_thursday_order_predictions(target_date_str)
            logging.info('处理周四订货建议 (周五-周六及下周一)')
            
        else:  # 其他日期（周二、周三、周五、周六、周日）- 服务不可用
            response_data = {
                'message': '该服务仅在周一与周四使用',
                'target_date': target_date_str,
                'day_of_week': day_name,
                'status': 'service_unavailable',
                'available_days': ['周一', '周四'],
                'system_version': 'enhanced_predictor',
                'beer_purchase_recommendations': {},
                'note': '周一可获取周二-周四的订货建议，周四可获取周五-周六及下周一的订货建议'
            }
            logging.info(f'非下单日 ({day_name})，返回服务不可用消息')
        
        logging.info(f'成功生成响应，类型: {response_data.get("type", "service_unavailable")}')
        
        return func.HttpResponse(
            json.dumps(response_data, ensure_ascii=False, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logging.error(f"增强版API处理错误: {e}")
        error_response = {
            'error': '增强版系统内部错误',
            'error_details': str(e),
            'error_code': 'ENHANCED_INTERNAL_ERROR',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'system_version': 'enhanced_predictor',
            'beer_purchase_recommendations': {}
        }
        
        return func.HttpResponse(
            json.dumps(error_response, ensure_ascii=False, indent=2),
            status_code=500,
            mimetype="application/json"
        )