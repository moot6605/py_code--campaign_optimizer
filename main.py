"""
MAIN PROGRAM - MARKETING CAMPAIGN OPTIMIZER
==========================================

Orchestrates full ML pipeline:
- Data processing
- Model training
- Business analysis
- Visualization
- Reporting

Usage:
    python main.py [--data-file path/to/data.csv] [--save-models] [--generate-report]
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Import local modules
from src.config import config
from src.data_processor import DataProcessor
from src.ml_models import MLModelManager
from src.business_analytics import BusinessAnalytics
from src.visualization import CampaignVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('campaign_optimizer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CampaignOptimizerPipeline:
    """
    Main class for complete ML pipeline
    """
    
    def __init__(self):
        """Initialize pipeline components"""
        self.data_processor = DataProcessor()
        self.ml_manager = MLModelManager()
        self.business_analytics = BusinessAnalytics()
        self.visualizer = CampaignVisualizer()
        
        # Pipeline state
        self.pipeline_results = {}
        self.execution_time = {}
        
    def run_full_pipeline(self, data_file: str = None, save_models: bool = False, 
                         generate_report: bool = False) -> Dict[str, Any]:
        """
        Execute complete ML pipeline
        
        Args:
            data_file: Path to data file
            save_models: Whether to save trained models
            generate_report: Whether to generate HTML report
            
        Returns:
            Dict with pipeline results
        """
        logger.info("=" * 60)
        logger.info("MARKETING KAMPAGNE OPTIMIZER - PIPELINE START")
        logger.info("=" * 60)
        
        pipeline_start = datetime.now()
        
        try:
            # 1. Data processing
            logger.info("Step 1: Data processing")
            step_start = datetime.now()
            
            pipeline_result = self.data_processor.process_pipeline(data_file)
            if pipeline_result is None:
                raise Exception("Data processing failed")
            
            df_processed, X, feature_names = pipeline_result
            self.execution_time['data_processing'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Data processing completed ({self.execution_time['data_processing']:.2f}s)")
            
            # 2. Customer segmentation
            logger.info("Step 2: Customer segmentation")
            step_start = datetime.now()
            
            df_segmented = self.ml_manager.perform_clustering(df_processed)
            self.execution_time['clustering'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Customer segmentation completed ({self.execution_time['clustering']:.2f}s)")
            
            # 3. ML model training
            logger.info("Step 3: ML model training")
            step_start = datetime.now()
            
            y = df_segmented['Antwort_Letzte_Kampagne']
            model_results = self.ml_manager.train_classification_models(X, y)
            self.execution_time['model_training'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Model training completed ({self.execution_time['model_training']:.2f}s)")
            
            # 4. Predict conversion probabilities
            logger.info("Step 4: Conversion predictions")
            step_start = datetime.now()
            
            conversion_probabilities = self.ml_manager.predict_conversion_probability(X)
            df_segmented['Conversion_Wahrscheinlichkeit'] = conversion_probabilities
            self.execution_time['predictions'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Predictions completed ({self.execution_time['predictions']:.2f}s)")
            
            # 5. Business analysis
            logger.info("Step 5: Business analysis and ROI calculation")
            step_start = datetime.now()
            
            # Calculate ROI scenarios
            roi_scenarios = self.business_analytics.calculate_roi_scenarios(df_segmented)
            
            # Analyze campaign performance
            campaign_performance = self.business_analytics.analyze_campaign_performance(df_segmented)
            
            # Segment analysis
            segment_analysis = self.business_analytics.segment_analysis(df_segmented)
            
            # Executive Summary
            executive_summary = self.business_analytics.generate_executive_summary(df_segmented, roi_scenarios)
            
            self.execution_time['business_analysis'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Business analysis completed ({self.execution_time['business_analysis']:.2f}s)")
            
            # 6. Create visualizations
            logger.info("Step 6: Creating visualizations")
            step_start = datetime.now()
            
            dashboard_plots = self.visualizer.create_executive_dashboard(
                df_segmented, roi_scenarios, segment_analysis
            )
            
            self.execution_time['visualization'] = (datetime.now() - step_start).total_seconds()
            logger.info(f"Visualizations created ({self.execution_time['visualization']:.2f}s)")
            
            # 7. Save models (optional)
            if save_models:
                logger.info("Step 7: Saving models")
                step_start = datetime.now()
                
                saved_models = self.ml_manager.save_models()
                self.execution_time['model_saving'] = (datetime.now() - step_start).total_seconds()
                logger.info(f"Models saved ({self.execution_time['model_saving']:.2f}s)")
            else:
                saved_models = {}
            
            # 8. Generate report (optional)
            if generate_report:
                logger.info("Step 8: Generating report")
                step_start = datetime.now()
                
                report_path = self._generate_comprehensive_report(
                    df_segmented, roi_scenarios, campaign_performance, 
                    segment_analysis, executive_summary, model_results
                )
                
                self.execution_time['report_generation'] = (datetime.now() - step_start).total_seconds()
                logger.info(f"Report generated: {report_path} ({self.execution_time['report_generation']:.2f}s)")
            else:
                report_path = None
            
            # Compile pipeline results
            total_time = (datetime.now() - pipeline_start).total_seconds()
            
            self.pipeline_results = {
                'data': df_segmented,
                'ml_features': X,
                'feature_names': feature_names,
                'model_results': model_results,
                'roi_scenarios': roi_scenarios,
                'campaign_performance': campaign_performance,
                'segment_analysis': segment_analysis,
                'executive_summary': executive_summary,
                'dashboard_plots': dashboard_plots,
                'saved_models': saved_models,
                'report_path': report_path,
                'execution_time': self.execution_time,
                'total_execution_time': total_time
            }
            
            logger.info("=" * 60)
            logger.info("PIPELINE ERFOLGREICH ABGESCHLOSSEN")
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info("=" * 60)
            
            # Log brief summary
            self._log_pipeline_summary()
            
            return self.pipeline_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline error: {str(e)}")
            raise
    
    def _log_pipeline_summary(self) -> None:
        """Log pipeline results summary"""
        if not self.pipeline_results:
            return
        
        summary = self.pipeline_results['executive_summary']
        best_strategy = summary['best_strategy']
        
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE ZUSAMMENFASSUNG")
        logger.info("=" * 50)
        logger.info(f"Total customers: {summary['total_customers']:,}")
        logger.info(f"High-potential customers: {summary['high_potential_customers']:,} ({summary['high_potential_percentage']:.1f}%)")
        logger.info(f"Avg conversion probability: {summary['avg_conversion_probability']:.1%}")
        logger.info(f"Best strategy: {best_strategy['name']}")
        logger.info(f"Expected ROI: {best_strategy['roi']:.1f}%")
        logger.info(f"Expected profit: {best_strategy['expected_profit']:,.2f}€")
        logger.info(f"Estimated annual impact: {summary['annual_impact_estimate']:,.2f}€")
        logger.info("=" * 50)
    
    def _generate_comprehensive_report(self, df: Any, roi_scenarios: Any, campaign_performance: Any,
                                     segment_analysis: Any, executive_summary: Any, model_results: Any) -> str:
        """
        Generate comprehensive HTML report
        
        Returns:
            Path to generated report
        """
        from datetime import datetime
        import os
        
        # Create report directory
        report_dir = "reports"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"marketing_campaign_report_{timestamp}.html"
        report_path = os.path.join(report_dir, report_filename)
        
        # Generate HTML report
        html_content = self._create_html_report(
            df, roi_scenarios, campaign_performance, segment_analysis, 
            executive_summary, model_results
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return report_path
    
    def _create_html_report(self, df: Any, roi_scenarios: Any, campaign_performance: Any,
                           segment_analysis: Any, executive_summary: Any, model_results: Any) -> str:
        """Create HTML report content"""
        
        best_strategy = executive_summary['best_strategy']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="de">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Marketing Kampagne Optimizer - Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: #1f77b4; color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
                .success {{ color: #28a745; font-weight: bold; }}
                .warning {{ color: #ffc107; font-weight: bold; }}
                .error {{ color: #dc3545; font-weight: bold; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Marketing Campaign Optimizer</h1>
                <h2>Comprehensive Analysis Report</h2>
                <p>Generated: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>🎯 Executive Summary</h2>
                <div class="metric">
                    <strong>Total customers:</strong> {executive_summary['total_customers']:,}
                </div>
                <div class="metric">
                    <strong>High-potential customers:</strong> {executive_summary['high_potential_customers']:,} 
                    ({executive_summary['high_potential_percentage']:.1f}%)
                </div>
                <div class="metric">
                    <strong>Avg conversion probability:</strong> {executive_summary['avg_conversion_probability']:.1%}
                </div>
                
                <h3>🏆 Best strategy: {best_strategy['name']}</h3>
                <ul>
                    <li><strong>ROI:</strong> <span class="{'success' if best_strategy['roi'] > 0 else 'error'}">{best_strategy['roi']:.1f}%</span></li>
                    <li><strong>Expected profit:</strong> {best_strategy['expected_profit']:,.2f}€</li>
                    <li><strong>Customers to target:</strong> {best_strategy['customers_to_target']:,}</li>
                    <li><strong>Estimated annual impact:</strong> {executive_summary['annual_impact_estimate']:,.2f}€</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>💰 ROI Scenario Comparison</h2>
                <table>
                    <tr>
                        <th>Scenario</th>
                        <th>Customer Count</th>
                        <th>Expected Conversions</th>
                        <th>Conversion Rate (%)</th>
                        <th>ROI (%)</th>
                        <th>Expected Profit (€)</th>
                    </tr>
        """
        
        # ROI scenarios table
        for _, row in roi_scenarios.iterrows():
            roi_class = 'success' if row['ROI'] > 0 else 'error'
            html += f"""
                    <tr>
                        <td>{row['Szenario']}</td>
                        <td>{row['Anzahl_Kunden']:,}</td>
                        <td>{row['Erwartete_Conversions']}</td>
                        <td>{row['Conversion_Rate']}</td>
                        <td class="{roi_class}">{row['ROI']}</td>
                        <td>{row['Erwarteter_Gewinn']:,.2f}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>👥 Customer Segment Analysis</h2>
                <table>
                    <tr>
                        <th>Segment</th>
                        <th>Customer Count</th>
                        <th>Avg Conversion Prob.</th>
                        <th>Avg Spending (€)</th>
                        <th>Avg Engagement</th>
                    </tr>
        """
        
        # Segment analysis table
        for segment, row in segment_analysis.iterrows():
            html += f"""
                    <tr>
                        <td>{segment}</td>
                        <td>{row['Anzahl_Kunden']:,}</td>
                        <td>{row['Ø_Conversion_Prob']:.3f}</td>
                        <td>{row['Ø_Ausgaben']:,.2f}</td>
                        <td>{row['Ø_Engagement']:.2f}</td>
                    </tr>
            """
        
        html += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>🔧 Technical Details</h2>
                <h3>Model Performance</h3>
                <ul>
                    <li><strong>Best model:</strong> {self.ml_manager.best_model_name}</li>
                    <li><strong>F1-Score:</strong> {model_results[self.ml_manager.best_model_name]['f1_score']:.4f}</li>
                    <li><strong>ROC-AUC:</strong> {model_results[self.ml_manager.best_model_name]['roc_auc']:.4f}</li>
                    <li><strong>Accuracy:</strong> {model_results[self.ml_manager.best_model_name]['accuracy']:.4f}</li>
                </ul>
                
                <h3>Pipeline Execution Times</h3>
                <ul>
        """
        
        # Execution times
        for step, time_taken in self.execution_time.items():
            html += f"<li><strong>{step.replace('_', ' ').title()}:</strong> {time_taken:.2f}s</li>"
        
        html += f"""
                    <li><strong>Total time:</strong> {self.pipeline_results['total_execution_time']:.2f}s</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>📋 Action Recommendations</h2>
                <ol>
        """
        
        # Recommendations
        for recommendation in executive_summary['key_recommendations']:
            html += f"<li>{recommendation}</li>"
        
        html += """
                </ol>
            </div>
            
            <footer style="margin-top: 50px; padding: 20px; background: #f8f9fa; border-radius: 8px;">
                <p><strong>Marketing Campaign Optimizer</strong> - ML Final Project</p>
                <p>Generated with Python, Scikit-learn, and Plotly</p>
            </footer>
        </body>
        </html>
        """
        
        return html

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Marketing Kampagne Optimizer')
    parser.add_argument('--data-file', type=str, help='Path to CSV data file')
    parser.add_argument('--save-models', action='store_true', help='Save trained models')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    parser.add_argument('--campaign-cost', type=float, default=5.0, help='Campaign cost per customer')
    parser.add_argument('--revenue-per-conversion', type=float, default=150.0, help='Revenue per conversion')
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = CampaignOptimizerPipeline()
        
        # Set business parameters
        pipeline.business_analytics.set_business_parameters(
            args.campaign_cost, 
            args.revenue_per_conversion
        )
        
        # Execute pipeline
        results = pipeline.run_full_pipeline(
            data_file=args.data_file,
            save_models=args.save_models,
            generate_report=args.generate_report
        )
        
        print("\n" + "="*60)
        print("PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
        print("="*60)
        print(f"Total time: {results['total_execution_time']:.2f} seconds")
        
        if args.generate_report and results['report_path']:
            print(f"Report generated: {results['report_path']}")
        
        if args.save_models and results['saved_models']:
            print(f"Models saved: {list(results['saved_models'].keys())}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())