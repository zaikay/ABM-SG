# analysis/visualizers.py V4

"""
Visualization components for the prosumer simulation.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

class SimulationVisualizer:
    """
    Creates visualizations for simulation results.
    """
    def __init__(self, metrics_analyzer):
        """
        Initialize the visualization components.
        
        Args:
            metrics_analyzer: MetricsAnalyzer instance
        """
        self.metrics = metrics_analyzer
        
        # Set up plot style
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """
        Set up the plot style for consistent visualizations.
        """
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["font.size"] = 12
    
    def plot_adoption_curve(self, output_dir="results"):
        """
        Plot the adoption curve over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        adoption_data = self.metrics.get_adoption_curve()
        
        if adoption_data.empty:
            return
        
        plt.figure()
        sns.lineplot(x="Year", y="AdoptionRate", data=adoption_data, marker="o", linewidth=2)
        plt.title("Solar PV Adoption Rate Over Time")
        plt.xlabel("Year")
        plt.ylabel("Adoption Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/adoption_curve.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_fossil_dependency(self, output_dir="results"):
        """
        Plot fossil fuel dependency over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        fossil_data = self.metrics.get_fossil_dependency()
        
        if fossil_data.empty:
            return
        
        plt.figure()
        sns.lineplot(x="Year", y="FossilDependency", data=fossil_data, marker="o", linewidth=2, color="brown")
        plt.title("Fossil Fuel Dependency Over Time")
        plt.xlabel("Year")
        plt.ylabel("Fossil Dependency")
        plt.ylim(0, 1)
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/fossil_dependency.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_energy_costs(self, output_dir="results"):
        """
        Plot energy costs over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        cost_data = self.metrics.get_energy_costs()
        
        if cost_data.empty:
            return
        
        plt.figure()
        sns.lineplot(x="Year", y="MeanCost", data=cost_data, marker="o", linewidth=2, label="Average")
        sns.lineplot(x="Year", y="ProsumerCost", data=cost_data, marker="s", linewidth=2, label="Prosumers")
        sns.lineplot(x="Year", y="NonProsumerCost", data=cost_data, marker="^", linewidth=2, label="Non-Prosumers")
        
        plt.title("Monthly Energy Costs Over Time")
        plt.xlabel("Year")
        plt.ylabel("Monthly Cost ($)")
        plt.grid(True)
        plt.legend()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/energy_costs.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_income_class_adoption(self, output_dir="results"):
        """
        Plot adoption by income class.
        
        Args:
            output_dir: Directory to save the plot
        """
        class_data = self.metrics.get_income_class_adoption()
        
        if class_data.empty:
            return
        
        plt.figure()
        sns.lineplot(x="Year", y="AdoptionRate", hue="Class", data=class_data, marker="o", linewidth=2)
        
        plt.title("Solar PV Adoption by Income Class")
        plt.xlabel("Year")
        plt.ylabel("Adoption Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend(title="Income Class")
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/class_adoption.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def plot_final_adoption_distribution(self, output_dir="results"):
        """
        Plot the final adoption distribution by income class.
        
        Args:
            output_dir: Directory to save the plot
        """
        class_data = self.metrics.get_income_class_adoption()
        
        if class_data.empty:
            return
        
        # Get the last year's data
        last_year = class_data["Year"].max()
        final_data = class_data[class_data["Year"] == last_year]
        
        plt.figure()
        sns.barplot(x="Class", y="AdoptionRate", data=final_data, palette="viridis")
        
        plt.title(f"Final Solar PV Adoption by Income Class (Year {last_year})")
        plt.xlabel("Income Class")
        plt.ylabel("Adoption Rate")
        plt.ylim(0, 1)
        plt.grid(True, axis="y")
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/final_class_adoption.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    def create_all_visualizations(self, output_dir="results"):
        """
        Create all standard visualizations.
        
        Args:
            output_dir: Directory to save the plots
        """
        self.plot_adoption_curve(output_dir)
        self.plot_fossil_dependency(output_dir)
        self.plot_energy_costs(output_dir)
        self.plot_income_class_adoption(output_dir)
        self.plot_final_adoption_distribution(output_dir)
        
        # Create new macro-level visualizations
        self.plot_peak_load_evolution(output_dir)
        self.plot_credit_utilization_evolution(output_dir)
        self.plot_seasonal_stress_patterns(output_dir)
        self.plot_npv_statistical_summary(output_dir)
        self.plot_npv_income_analysis(output_dir) 
        self.plot_payback_period_trends(output_dir)

        # Add the new visualizations
        self.plot_energy_balance_evolution(output_dir)
        self.plot_grid_dependency_reduction(output_dir)
        self.plot_adoption_velocity_by_class(output_dir)
        self.plot_energy_cost_distribution_evolution(output_dir)
        self.plot_monthly_consumption_generation_trends(output_dir)
        
        print(f"✅ All visualizations completed and saved to {output_dir}")

                # Create subfolder for macro analysis
        macro_dir = f"{output_dir}/macro_analysis"
        os.makedirs(macro_dir, exist_ok=True)
        
        # Copy macro plots to subfolder
        import shutil
        macro_plots = [
            "peak_load_evolution", "credit_utilization_evolution", 
            "seasonal_stress_patterns", "npv_distribution_evolution", 
            "payback_period_trends"
        ]
        
        for plot_name in macro_plots:
            src = f"{output_dir}/{plot_name}.png"
            dst = f"{macro_dir}/{plot_name}.png"
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        print(f"📊 Created macro-level visualizations in {macro_dir}")

    def plot_potential_vs_actual_adoption(self, agent_data, output_dir="results"):
        """
        Plot comparison between potential and actual adoption rates.
        FIXED to work with agent_data instead of model reference.
        """
        if agent_data.empty:
            return
        
        # Filter to get household data only
        household_data = agent_data[agent_data['AgentType'] == 'Household'].copy()
        
        if household_data.empty:
            return
        
        # Get final step data
        final_step = household_data['Step'].max()
        final_data = household_data[household_data['Step'] == final_step]
        
        # Calculate adoption rates by income class
        adoption_by_class = final_data.groupby('IncomeClass').agg({
            'IsProsumer': 'mean',
            'IsPotentialProsumer': 'mean'
        }).reset_index()
        
        adoption_by_class.columns = ['Income Class', 'Actual Adoption Rate', 'Potential Adoption Rate']
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(adoption_by_class))
        width = 0.35
        
        plt.bar(x - width/2, adoption_by_class['Actual Adoption Rate'] * 100, width, label='Actual Adoption')
        plt.bar(x + width/2, adoption_by_class['Potential Adoption Rate'] * 100, width, label='Potential Adoption')
        
        plt.xlabel('Income Class')
        plt.ylabel('Adoption Rate (%)')
        plt.title('Actual vs. Potential Adoption Rates by Income Class')
        plt.xticks(x, adoption_by_class['Income Class'])
        plt.legend()
        plt.grid(True, axis='y')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/potential_vs_actual_adoption.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ADD these methods to the existing SimulationVisualizer class:

    def plot_peak_load_evolution(self, output_dir="results"):
        """
        Plot exact peak load evolution over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        peak_data = self.metrics.get_peak_load_evolution()
        
        if peak_data.empty:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Main plot: Peak load over time
        plt.subplot(2, 1, 1)
        sns.lineplot(x="Year", y="PeakLoad", data=peak_data, marker="o", linewidth=2)
        plt.title("Grid Peak Load Evolution (Exact Calculation)")
        plt.ylabel("Peak Load (kW)")
        plt.grid(True)
        
        # Secondary plot: Seasonal breakdown
        plt.subplot(2, 1, 2)
        sns.boxplot(x="Season", y="PeakLoad", data=peak_data)
        plt.title("Peak Load Distribution by Season")
        plt.ylabel("Peak Load (kW)")
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/peak_load_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_credit_utilization_evolution(self, output_dir="results"):
        """
        Plot credit system utilization over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        credit_data = self.metrics.get_credit_utilization_evolution()
        
        if credit_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Credits earned vs used
        axes[0, 0].plot(credit_data["Year"], credit_data["CreditsEarned"], 
                    label="Credits Earned", linewidth=2, color='blue')
        axes[0, 0].plot(credit_data["Year"], credit_data["CreditsUsed"], 
                    label="Credits Used", linewidth=2, color='orange')
        axes[0, 0].set_title("Credit System Activity")
        axes[0, 0].set_ylabel("Energy Credits (kWh)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Utilization rate
        axes[0, 1].plot(credit_data["Year"], credit_data["UtilizationRate"], 
                    linewidth=2, color='green', marker='o', markersize=3)
        axes[0, 1].set_title("Credit Utilization Rate")
        axes[0, 1].set_ylabel("Utilization Rate")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Plot 3: Credits expired (waste)
        axes[1, 0].plot(credit_data["Year"], credit_data["CreditsExpired"], 
                    linewidth=2, color='red', marker='s', markersize=3)
        axes[1, 0].set_title("Credits Expired (Unused)")
        axes[1, 0].set_ylabel("Expired Credits (kWh)")
        axes[1, 0].grid(True)
        
        # Plot 4: Waste rate
        axes[1, 1].plot(credit_data["Year"], credit_data["WasteRate"], 
                    linewidth=2, color='purple', marker='^', markersize=3)
        axes[1, 1].set_title("Credit Waste Rate")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Waste Rate")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/credit_utilization_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_seasonal_stress_patterns(self, output_dir="results"):
        """
        Plot seasonal grid stress patterns.
        
        Args:
            output_dir: Directory to save the plot
        """
        stress_data = self.metrics.get_seasonal_stress_patterns()
        
        if not isinstance(stress_data, dict) or stress_data.get("monthly_data", pd.DataFrame()).empty:
            return
        
        monthly_data = stress_data["monthly_data"]
        seasonal_stats = stress_data["seasonal_stats"]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Monthly stress evolution
        axes[0, 0].plot(monthly_data["Year"], monthly_data["GridStressIndex"], 
                    linewidth=1, alpha=0.7, color='navy')
        axes[0, 0].set_title("Grid Stress Index Evolution")
        axes[0, 0].set_ylabel("Stress Index")
        axes[0, 0].grid(True)
        
        # Plot 2: Seasonal stress patterns
        sns.boxplot(x="Season", y="GridStressIndex", data=monthly_data, ax=axes[0, 1])
        axes[0, 1].set_title("Stress Index by Season")
        axes[0, 1].set_ylabel("Stress Index")
        axes[0, 1].grid(True)
        
        # Plot 3: Seasonal stress evolution over years
        for season in seasonal_stats["Season"].unique():
            season_data = seasonal_stats[seasonal_stats["Season"] == season]
            axes[1, 0].plot(season_data["Year"], season_data["MeanStress"], 
                        label=season, linewidth=2, marker='o', markersize=4)
        axes[1, 0].set_title("Seasonal Stress Trends")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Mean Stress Index")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Stress variability by season
        axes[1, 1].bar(seasonal_stats.groupby("Season")["StdStress"].mean().index,
                    seasonal_stats.groupby("Season")["StdStress"].mean().values)
        axes[1, 1].set_title("Stress Variability by Season")
        axes[1, 1].set_xlabel("Season")
        axes[1, 1].set_ylabel("Average Stress Std Dev")
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/seasonal_stress_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_npv_statistical_summary(self, output_dir="results"):
        """
        Plot NPV distribution evolution over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        npv_data = self.metrics.get_npv_statistics_by_year()
        
        if npv_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean and median NPV evolution
        axes[0, 0].plot(npv_data["Year"], npv_data["MeanNPV"], 
                    label="Mean NPV", linewidth=2, color='blue')
        axes[0, 0].plot(npv_data["Year"], npv_data["MedianNPV"], 
                    label="Median NPV", linewidth=2, color='orange')
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, label="Break-even")
        axes[0, 0].set_title("NPV Central Tendency Evolution")
        axes[0, 0].set_ylabel("NPV ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: NPV distribution spread
        axes[0, 1].fill_between(npv_data["Year"], npv_data["Q1"], npv_data["Q3"], 
                            alpha=0.3, label="IQR", color='green')
        axes[0, 1].plot(npv_data["Year"], npv_data["MedianNPV"], 
                    linewidth=2, color='darkgreen', label="Median")
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title("NPV Distribution Spread")
        axes[0, 1].set_ylabel("NPV ($)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Positive NPV rate
        axes[1, 0].plot(npv_data["Year"], npv_data["PositiveNPVRate"], 
                    linewidth=2, color='purple', marker='o', markersize=4)
        axes[1, 0].set_title("Fraction with Positive NPV")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Positive NPV Rate")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True)
        
        # Plot 4: NPV standard deviation
        axes[1, 1].plot(npv_data["Year"], npv_data["StdNPV"], 
                    linewidth=2, color='brown', marker='s', markersize=4)
        axes[1, 1].set_title("NPV Variability")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("NPV Standard Deviation ($)")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/npv_distribution_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_payback_period_trends(self, output_dir="results"):
        """
        Plot payback period trends by income class.
        
        Args:
            output_dir: Directory to save the plot
        """
        payback_data = self.metrics.get_payback_period_trends()
        
        if payback_data.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Mean payback by income class over time
        for income_class in sorted(payback_data["IncomeClass"].unique()):
            class_data = payback_data[payback_data["IncomeClass"] == income_class]
            axes[0, 0].plot(class_data["Year"], class_data["MeanPayback"], 
                        label=f"Class {income_class}", linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_title("Mean Payback Period by Income Class")
        axes[0, 0].set_ylabel("Payback Period (years)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Median payback by income class
        for income_class in sorted(payback_data["IncomeClass"].unique()):
            class_data = payback_data[payback_data["IncomeClass"] == income_class]
            axes[0, 1].plot(class_data["Year"], class_data["MedianPayback"], 
                        label=f"Class {income_class}", linewidth=2, marker='s', markersize=3)
        axes[0, 1].set_title("Median Payback Period by Income Class")
        axes[0, 1].set_ylabel("Payback Period (years)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Final year payback comparison
        final_year = payback_data["Year"].max()
        final_data = payback_data[payback_data["Year"] == final_year]
        axes[1, 0].bar(final_data["IncomeClass"], final_data["MeanPayback"])
        axes[1, 0].set_title(f"Final Payback Periods by Income Class (Year {final_year})")
        axes[1, 0].set_xlabel("Income Class")
        axes[1, 0].set_ylabel("Mean Payback Period (years)")
        axes[1, 0].grid(True, axis='y')
        
        # Plot 4: Adoption rate vs payback period correlation
        axes[1, 1].scatter(payback_data["MeanPayback"], payback_data["AdoptionRate"], 
                        c=payback_data["IncomeClass"], cmap='viridis', alpha=0.6)
        axes[1, 1].set_title("Adoption Rate vs Payback Period")
        axes[1, 1].set_xlabel("Mean Payback Period (years)")
        axes[1, 1].set_ylabel("Adoption Rate")
        axes[1, 1].grid(True)
        
        # Add colorbar for income class
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label("Income Class")
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/payback_period_trends.png", dpi=300, bbox_inches="tight")
        plt.close()


    def plot_energy_balance_evolution(self, output_dir="results"):
        """
        Plot energy balance evolution showing consumption vs generation.
        
        Args:
            output_dir: Directory to save the plot
        """
        if not all(col in self.metrics.model_data.columns for col in 
                ["TotalConsumption", "rational_TotalGeneration", "rational_GridConsumption"]):
            print("Warning: Energy balance data not available")
            return

        energy_data = self.metrics.model_data[["Year", "Month", "TotalConsumption", 
                                            "rational_TotalGeneration", "rational_GridConsumption"]].copy()

        # Rename for compatibility with existing plot code
        energy_data = energy_data.rename(columns={
            'rational_TotalGeneration': 'TotalGeneration',
            'rational_GridConsumption': 'GridConsumption'
        })
        # Calculate derived metrics
        energy_data["SolarPenetration"] = energy_data["TotalGeneration"] / energy_data["TotalConsumption"]
        energy_data["SelfSufficiency"] = 1 - (energy_data["GridConsumption"] / energy_data["TotalConsumption"])
        energy_data["SolarPenetration"] = energy_data["SolarPenetration"].fillna(0)
        energy_data["SelfSufficiency"] = energy_data["SelfSufficiency"].fillna(0)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Total consumption vs generation
        axes[0, 0].plot(energy_data["Year"], energy_data["TotalConsumption"], 
                        label="Total Consumption", linewidth=2, color='red')
        axes[0, 0].plot(energy_data["Year"], energy_data["TotalGeneration"], 
                        label="Solar Generation", linewidth=2, color='green')
        axes[0, 0].set_title("System Energy Balance")
        axes[0, 0].set_ylabel("Energy (kWh)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Solar penetration rate
        axes[0, 1].plot(energy_data["Year"], energy_data["SolarPenetration"], 
                        linewidth=2, color='orange', marker='o', markersize=3)
        axes[0, 1].set_title("Solar Penetration Rate")
        axes[0, 1].set_ylabel("Solar Generation / Total Consumption")
        axes[0, 1].set_ylim(0, max(1, energy_data["SolarPenetration"].max() * 1.1))
        axes[0, 1].grid(True)
        
        # Plot 3: Grid consumption evolution
        axes[1, 0].plot(energy_data["Year"], energy_data["GridConsumption"], 
                        linewidth=2, color='brown', marker='s', markersize=3)
        axes[1, 0].set_title("Grid Consumption Evolution")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Grid Consumption (kWh)")
        axes[1, 0].grid(True)
        
        # Plot 4: System self-sufficiency
        axes[1, 1].plot(energy_data["Year"], energy_data["SelfSufficiency"], 
                        linewidth=2, color='purple', marker='^', markersize=3)
        axes[1, 1].set_title("System Self-Sufficiency")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Self-Sufficiency Rate")
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/energy_balance_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()


    def plot_grid_dependency_reduction(self, output_dir="results"):
        """
        Plot grid dependency reduction over time with multiple perspectives.
        
        Args:
            output_dir: Directory to save the plot
        """
        if "FossilDependency" not in self.metrics.model_data.columns:
            print("Warning: Fossil dependency data not available")
            return
        
        # Extract fossil dependency data
        dependency_data = self.metrics.model_data[["Year", "Month", "FossilDependency"]].copy()
        
        # Calculate reduction metrics
        initial_dependency = dependency_data["FossilDependency"].iloc[0]
        dependency_data["DependencyReduction"] = initial_dependency - dependency_data["FossilDependency"]
        dependency_data["ReductionPercent"] = (dependency_data["DependencyReduction"] / initial_dependency) * 100
        
        # Calculate rolling averages
        dependency_data["FossilDependency_MA12"] = dependency_data["FossilDependency"].rolling(12, min_periods=1).mean()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Fossil dependency over time
        axes[0, 0].plot(dependency_data["Year"], dependency_data["FossilDependency"], 
                        linewidth=1, alpha=0.7, color='red', label='Monthly')
        axes[0, 0].plot(dependency_data["Year"], dependency_data["FossilDependency_MA12"], 
                        linewidth=3, color='darkred', label='12-Month Average')
        axes[0, 0].set_title("Fossil Fuel Dependency Over Time")
        axes[0, 0].set_ylabel("Fossil Dependency")
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Absolute reduction
        axes[0, 1].fill_between(dependency_data["Year"], 0, dependency_data["DependencyReduction"], 
                            alpha=0.6, color='green')
        axes[0, 1].plot(dependency_data["Year"], dependency_data["DependencyReduction"], 
                        linewidth=2, color='darkgreen')
        axes[0, 1].set_title("Absolute Dependency Reduction")
        axes[0, 1].set_ylabel("Dependency Reduction")
        axes[0, 1].grid(True)
        
        # Plot 3: Percentage reduction
        axes[1, 0].plot(dependency_data["Year"], dependency_data["ReductionPercent"], 
                        linewidth=2, color='blue', marker='o', markersize=3)
        axes[1, 0].set_title("Percentage Dependency Reduction")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Reduction (%)")
        axes[1, 0].grid(True)
        
        # Plot 4: Reduction rate (derivative)
        dependency_data["ReductionRate"] = dependency_data["ReductionPercent"].diff().rolling(6, min_periods=1).mean()
        axes[1, 1].plot(dependency_data["Year"], dependency_data["ReductionRate"], 
                        linewidth=2, color='purple', marker='s', markersize=3)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        axes[1, 1].set_title("Rate of Dependency Reduction")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Reduction Rate (%/year)")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/grid_dependency_reduction.png", dpi=300, bbox_inches="tight")
        plt.close()


    def plot_adoption_velocity_by_class(self, output_dir="results"):
        """
        Plot adoption velocity (rate of change) by income class.
        Fixed version that handles duplicate data properly.
        """
        class_data = self.metrics.get_income_class_adoption()
        
        if class_data.empty:
            print("Warning: Income class adoption data not available")
            return
        
        try:
            # Remove duplicates by taking the last value for each Year-Class combination
            class_data_clean = class_data.groupby(["Year", "Class"]).last().reset_index()
            
            # Calculate adoption velocity (rate of change) for each class
            velocity_data = []
            
            for income_class in class_data_clean["Class"].unique():
                class_subset = class_data_clean[class_data_clean["Class"] == income_class].sort_values("Year")
                class_subset = class_subset.reset_index(drop=True)
                
                # Calculate velocity and acceleration
                class_subset["AdoptionVelocity"] = class_subset["AdoptionRate"].diff()
                class_subset["AdoptionAcceleration"] = class_subset["AdoptionVelocity"].diff()
                
                velocity_data.append(class_subset)
            
            if not velocity_data:
                print("Warning: No velocity data calculated")
                return
            
            combined_velocity = pd.concat(velocity_data, ignore_index=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Adoption rate by class
            sns.lineplot(x="Year", y="AdoptionRate", hue="Class", data=class_data_clean, 
                        ax=axes[0, 0], marker="o", linewidth=2)
            axes[0, 0].set_title("Adoption Rate by Income Class")
            axes[0, 0].set_ylabel("Adoption Rate")
            axes[0, 0].grid(True)
            
            # Plot 2: Adoption velocity by class (with NaN filtering)
            velocity_clean = combined_velocity.dropna(subset=['AdoptionVelocity'])
            if not velocity_clean.empty:
                sns.lineplot(x="Year", y="AdoptionVelocity", hue="Class", data=velocity_clean, 
                            ax=axes[0, 1], marker="s", linewidth=2)
            axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[0, 1].set_title("Adoption Velocity by Income Class")
            axes[0, 1].set_ylabel("Adoption Rate Change (per year)")
            axes[0, 1].grid(True)
            
            # Plot 3: Alternative cumulative adoption (using line plots instead of stackplot)
            for income_class in class_data_clean["Class"].unique():
                class_subset = class_data_clean[class_data_clean["Class"] == income_class]
                axes[1, 0].plot(class_subset["Year"], class_subset["AdoptionRate"], 
                            label=income_class, linewidth=2, marker='o')
            
            axes[1, 0].set_title("Adoption Rate Comparison by Class")
            axes[1, 0].set_xlabel("Year")
            axes[1, 0].set_ylabel("Adoption Rate")
            axes[1, 0].legend(loc='upper left')
            axes[1, 0].grid(True)
            
            # Plot 4: Adoption acceleration by class (with NaN filtering)
            acceleration_clean = combined_velocity.dropna(subset=['AdoptionAcceleration'])
            if not acceleration_clean.empty:
                sns.lineplot(x="Year", y="AdoptionAcceleration", hue="Class", data=acceleration_clean, 
                            ax=axes[1, 1], marker="^", linewidth=2)
            axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            axes[1, 1].set_title("Adoption Acceleration by Income Class")
            axes[1, 1].set_xlabel("Year")
            axes[1, 1].set_ylabel("Acceleration (change in velocity)")
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(f"{output_dir}/adoption_velocity_by_class.png", dpi=300, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            print(f"Error in plot_adoption_velocity_by_class: {e}")
            # Create a simple fallback plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error creating adoption velocity plot:\n{str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Adoption Velocity by Income Class (Error)")
            plt.savefig(f"{output_dir}/adoption_velocity_by_class.png", dpi=300, bbox_inches="tight")
            plt.close()


    def plot_energy_cost_distribution_evolution(self, output_dir="results"):
        """
        Plot energy cost distribution evolution over time.
        
        Args:
            output_dir: Directory to save the plot
        """
        if "MonthlyCosts" not in self.metrics.model_data.columns:
            print("Warning: Monthly costs data not available")
            return
        
        # Extract cost data
        cost_evolution = []
        for idx, row in self.metrics.model_data.iterrows():
            year = row["Year"]
            costs = row["MonthlyCosts"]
            
            cost_evolution.append({
                "Year": year,
                "MeanCost": costs.get("mean", 0),
                "MedianCost": costs.get("median", 0),
                "MinCost": costs.get("min", 0),
                "MaxCost": costs.get("max", 0),
                "ProsumerMeanCost": costs.get("prosumer_mean", 0),
                "NonProsumerMeanCost": costs.get("nonprosumer_mean", 0)
            })
        
        cost_df = pd.DataFrame(cost_evolution)
        
        # Calculate cost inequality
        cost_df["CostInequality"] = cost_df["MaxCost"] / cost_df["MinCost"].replace(0, np.nan)
        cost_df["ProsumerSavings"] = cost_df["NonProsumerMeanCost"] - cost_df["ProsumerMeanCost"]
        cost_df["SavingsPercent"] = (cost_df["ProsumerSavings"] / cost_df["NonProsumerMeanCost"]) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cost distribution spread
        axes[0, 0].fill_between(cost_df["Year"], cost_df["MinCost"], cost_df["MaxCost"], 
                            alpha=0.3, color='lightblue', label='Min-Max Range')
        axes[0, 0].plot(cost_df["Year"], cost_df["MeanCost"], linewidth=2, color='blue', label='Mean')
        axes[0, 0].plot(cost_df["Year"], cost_df["MedianCost"], linewidth=2, color='navy', label='Median')
        axes[0, 0].set_title("Energy Cost Distribution Evolution")
        axes[0, 0].set_ylabel("Monthly Cost ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Prosumer vs Non-Prosumer costs
        axes[0, 1].plot(cost_df["Year"], cost_df["ProsumerMeanCost"], 
                        linewidth=2, color='green', marker='o', markersize=3, label='Prosumers')
        axes[0, 1].plot(cost_df["Year"], cost_df["NonProsumerMeanCost"], 
                        linewidth=2, color='red', marker='s', markersize=3, label='Non-Prosumers')
        axes[0, 1].set_title("Cost Comparison: Prosumers vs Non-Prosumers")
        axes[0, 1].set_ylabel("Monthly Cost ($)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Prosumer savings evolution
        axes[1, 0].fill_between(cost_df["Year"], 0, cost_df["ProsumerSavings"], 
                            alpha=0.6, color='lightgreen')
        axes[1, 0].plot(cost_df["Year"], cost_df["ProsumerSavings"], 
                        linewidth=2, color='darkgreen')
        axes[1, 0].set_title("Prosumer Monthly Savings")
        axes[1, 0].set_xlabel("Year")
        axes[1, 0].set_ylabel("Savings ($)")
        axes[1, 0].grid(True)
        
        # Plot 4: Savings percentage
        axes[1, 1].plot(cost_df["Year"], cost_df["SavingsPercent"], 
                        linewidth=2, color='purple', marker='^', markersize=3)
        axes[1, 1].set_title("Prosumer Savings Percentage")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Savings (%)")
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/energy_cost_distribution_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()


    def plot_monthly_consumption_generation_trends(self, output_dir="results"):
        """
        Plot monthly consumption and generation trends for different simulation years.
        FIXED: Updated to use scenario-specific columns (rational as default).
        """
        # CHANGE 1: Update column check to use rational scenario columns
        if not all(col in self.metrics.model_data.columns for col in 
                ["Year", "Month", "TotalConsumption", "rational_TotalGeneration", "CurrentMonthInYear"]):
            print("Warning: Monthly energy data not available")
            return
        
        # CHANGE 2: Extract monthly data with new column names
        monthly_data = self.metrics.model_data[["Year", "Month", "CurrentMonthInYear", 
                                            "TotalConsumption", "rational_TotalGeneration"]].copy()
        
        # CHANGE 3: Rename for compatibility with existing plot code
        monthly_data = monthly_data.rename(columns={'rational_TotalGeneration': 'TotalGeneration'})
        
        # Rest of the function remains EXACTLY the same...
        # Select key years for comparison
        available_years = monthly_data["Year"].unique()
        comparison_years = []
        for target_year in [1, 2, 5, 10, 15, 20]:
            if target_year in available_years:
                comparison_years.append(target_year)
        
        # Take first 4 available years if we don't have all target years
        if len(comparison_years) < 4:
            comparison_years = sorted(available_years)[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Monthly consumption patterns by year
        for year in comparison_years:
            year_data = monthly_data[monthly_data["Year"] == year]
            axes[0, 0].plot(year_data["CurrentMonthInYear"], year_data["TotalConsumption"], 
                        linewidth=2, marker='o', markersize=4, label=f'Year {year}')
        
        axes[0, 0].set_title("Monthly Consumption Patterns")
        axes[0, 0].set_xlabel("Month")
        axes[0, 0].set_ylabel("Total Consumption (kWh)")
        axes[0, 0].set_xticks(range(1, 13))
        axes[0, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Monthly generation patterns by year
        for year in comparison_years:
            year_data = monthly_data[monthly_data["Year"] == year]
            axes[0, 1].plot(year_data["CurrentMonthInYear"], year_data["TotalGeneration"], 
                        linewidth=2, marker='s', markersize=4, label=f'Year {year}')
        
        axes[0, 1].set_title("Monthly Solar Generation Patterns")
        axes[0, 1].set_xlabel("Month")
        axes[0, 1].set_ylabel("Total Generation (kWh)")
        axes[0, 1].set_xticks(range(1, 13))
        axes[0, 1].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Generation to consumption ratio by month and year
        monthly_data["GenConsRatio"] = monthly_data["TotalGeneration"] / monthly_data["TotalConsumption"]
        monthly_data["GenConsRatio"] = monthly_data["GenConsRatio"].fillna(0)
        
        for year in comparison_years:
            year_data = monthly_data[monthly_data["Year"] == year]
            axes[1, 0].plot(year_data["CurrentMonthInYear"], year_data["GenConsRatio"], 
                        linewidth=2, marker='^', markersize=4, label=f'Year {year}')
        
        axes[1, 0].set_title("Generation/Consumption Ratio by Month")
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel("Generation/Consumption Ratio")
        axes[1, 0].set_xticks(range(1, 13))
        axes[1, 0].set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 4: Heatmap of generation/consumption ratio evolution
        pivot_data = monthly_data.pivot(index="CurrentMonthInYear", columns="Year", values="GenConsRatio")
        
        # Select a subset of years for readability
        display_years = sorted(available_years)[::max(1, len(available_years)//10)]  # Show every nth year
        pivot_subset = pivot_data[display_years] if len(display_years) <= 10 else pivot_data.iloc[:, ::max(1, len(pivot_data.columns)//10)]
        
        im = axes[1, 1].imshow(pivot_subset.values, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        axes[1, 1].set_title("Generation/Consumption Ratio Heatmap")
        axes[1, 1].set_xlabel("Year")
        axes[1, 1].set_ylabel("Month")
        axes[1, 1].set_yticks(range(12))
        axes[1, 1].set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        axes[1, 1].set_xticks(range(len(pivot_subset.columns)))
        axes[1, 1].set_xticklabels(pivot_subset.columns, rotation=45)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[1, 1], label='Gen/Cons Ratio')
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/monthly_consumption_generation_trends.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_adoption_curve(self, output_dir="results"):
        """
        FIXED: Plot adoption curve using proper data extraction.
        """
        adoption_data = self.metrics.get_adoption_curve()
        
        if adoption_data.empty:
            print("⚠️ plot_adoption_curve: No adoption data available")
            return
        
        plt.figure(figsize=(12, 8))
        sns.lineplot(x="Year", y="AdoptionRate", data=adoption_data, marker="o", linewidth=2)
        plt.title("Solar PV Adoption Rate Over Time")
        plt.xlabel("Year")
        plt.ylabel("Adoption Rate")
        plt.ylim(0, 1)
        plt.grid(True)
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/adoption_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_final_adoption_distribution(self, output_dir="results"):
        """
        FIXED: Plot final adoption distribution by income class.
        """
        if self.metrics.agent_data.empty:
            print("⚠️ plot_final_adoption_distribution: No agent data available")
            return
        
        # Get final step data
        household_data = self.metrics.agent_data[self.metrics.agent_data['AgentType'] == 'Household']
        
        if household_data.empty:
            print("⚠️ plot_final_adoption_distribution: No household data available")
            return
        
        final_step = household_data['Step'].max()
        final_data = household_data[household_data['Step'] == final_step]
        
        if 'IncomeClass' not in final_data.columns:
            print("⚠️ plot_final_adoption_distribution: No income class data available")
            return
        
        # Calculate adoption rates by income class
        adoption_by_class = final_data.groupby('IncomeClass').agg({
            'IsProsumer': 'mean'
        }).reset_index()
        
        adoption_by_class.columns = ['Income Class', 'Adoption Rate']
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(adoption_by_class['Income Class'], adoption_by_class['Adoption Rate'] * 100)
        
        # Color bars by income class
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.xlabel('Income Class')
        plt.ylabel('Adoption Rate (%)')
        plt.title('Final Solar PV Adoption Rate by Income Class')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/final_adoption_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_credit_utilization_evolution(self, output_dir="results"):
        """
        FIXED: Plot credit system utilization over time.
        """
        credit_data = self.metrics.get_credit_utilization_evolution()
        
        if credit_data.empty:
            print("⚠️ plot_credit_utilization_evolution: No credit data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Credits earned vs used
        ax = axes[0, 0]
        ax.plot(credit_data["Year"], credit_data["CreditsEarned"], 
            label="Credits Earned", linewidth=2, marker="o")
        ax.plot(credit_data["Year"], credit_data["CreditsUsed"], 
            label="Credits Used", linewidth=2, marker="s")
        ax.set_title("Credit System: Earned vs Used")
        ax.set_ylabel("Credits")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Utilization rate
        ax = axes[0, 1]
        ax.plot(credit_data["Year"], credit_data["UtilizationRate"] * 100, 
            color="green", linewidth=2, marker="^")
        ax.set_title("Credit Utilization Rate")
        ax.set_ylabel("Utilization Rate (%)")
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Waste rate
        ax = axes[1, 0]
        ax.plot(credit_data["Year"], credit_data["WasteRate"] * 100, 
            color="red", linewidth=2, marker="v")
        ax.set_title("Credit Waste Rate")
        ax.set_ylabel("Waste Rate (%)")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Monthly breakdown
        ax = axes[1, 1]
        monthly_data = credit_data.groupby("Month").agg({
            "UtilizationRate": "mean"
        }).reset_index()
        ax.bar(monthly_data["Month"], monthly_data["UtilizationRate"] * 100)
        ax.set_title("Average Utilization by Month")
        ax.set_xlabel("Month")
        ax.set_ylabel("Utilization Rate (%)")
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/credit_utilization_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_seasonal_stress_patterns(self, output_dir="results"):
        """
        FIXED: Plot seasonal grid stress patterns.
        """
        stress_data = self.metrics.get_seasonal_stress_patterns()
        
        if stress_data.empty:
            print("⚠️ plot_seasonal_stress_patterns: No stress pattern data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Stress index over time
        ax = axes[0, 0]
        ax.plot(stress_data["Year"], stress_data["StressIndex"], 
            linewidth=2, marker="o", color="orange")
        ax.set_title("Grid Stress Index Evolution")
        ax.set_ylabel("Stress Index")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Seasonal breakdown
        ax = axes[0, 1]
        seasonal_data = stress_data.groupby("Season")["StressIndex"].mean().reset_index()
        bars = ax.bar(seasonal_data["Season"], seasonal_data["StressIndex"])
        ax.set_title("Average Stress by Season")
        ax.set_ylabel("Average Stress Index")
        ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Monthly heatmap
        ax = axes[1, 0]
        monthly_stress = stress_data.pivot_table(values="StressIndex", 
                                            index="Year", columns="Month", 
                                            aggfunc="mean")
        if not monthly_stress.empty:
            im = ax.imshow(monthly_stress.values, cmap="YlOrRd", aspect="auto")
            ax.set_title("Monthly Stress Heatmap")
            ax.set_xlabel("Month")
            ax.set_ylabel("Year")
            plt.colorbar(im, ax=ax)
        
        # Plot 4: Stress distribution
        ax = axes[1, 1]
        ax.hist(stress_data["StressIndex"], bins=20, alpha=0.7, color="red")
        ax.set_title("Stress Index Distribution")
        ax.set_xlabel("Stress Index")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/seasonal_stress_patterns.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_npv_income_analysis(self, output_dir="results"):
        """
        FIXED: Plot NPV distribution evolution over time.
        """
        npv_data = self.metrics.get_npv_raw_data()
        
        if npv_data.empty:
            print("⚠️ plot_npv_distribution_evolution: No NPV data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: NPV evolution by income class
        ax = axes[0, 0]
        if 'IncomeClass' in npv_data.columns:
            for income_class in sorted(npv_data['IncomeClass'].unique()):
                class_data = npv_data[npv_data['IncomeClass'] == income_class]
                yearly_avg = class_data.groupby('Year')['NPV'].mean()
                ax.plot(yearly_avg.index, yearly_avg.values, 
                    label=f'Class {income_class}', linewidth=2, marker='o')
            ax.legend()
        else:
            yearly_avg = npv_data.groupby('Year')['NPV'].mean()
            ax.plot(yearly_avg.index, yearly_avg.values, linewidth=2, marker='o')
        
        ax.set_title("NPV Evolution Over Time")
        ax.set_xlabel("Year")
        ax.set_ylabel("Average NPV ($)")
        ax.grid(True, alpha=0.3)
        
        # Plot 2: NPV distribution (final year)
        ax = axes[0, 1]
        final_year = npv_data['Year'].max()
        final_npv = npv_data[npv_data['Year'] == final_year]['NPV']
        ax.hist(final_npv, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Break-even')
        ax.set_title(f"NPV Distribution (Year {final_year})")
        ax.set_xlabel("NPV ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Positive NPV percentage over time
        ax = axes[1, 0]
        positive_npv_pct = npv_data.groupby('Year').apply(
            lambda x: (x['NPV'] > 0).mean() * 100
        )
        ax.plot(positive_npv_pct.index, positive_npv_pct.values, 
            linewidth=2, marker='s', color='green')
        ax.set_title("Households with Positive NPV (%)")
        ax.set_xlabel("Year")
        ax.set_ylabel("Percentage (%)")
        ax.grid(True, alpha=0.3)
        
        # Plot 4: NPV by adoption status
        ax = axes[1, 1]
        if 'IsProsumer' in npv_data.columns:
            final_data = npv_data[npv_data['Year'] == final_year]
            prosumer_npv = final_data[final_data['IsProsumer'] == True]['NPV']
            nonprosumer_npv = final_data[final_data['IsProsumer'] == False]['NPV']
            
            ax.hist(nonprosumer_npv, bins=20, alpha=0.6, label='Non-adopters', color='red')
            ax.hist(prosumer_npv, bins=20, alpha=0.6, label='Adopters', color='green')
            ax.axvline(0, color='black', linestyle='--', linewidth=2)
            ax.set_title("NPV by Adoption Status")
            ax.set_xlabel("NPV ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/npv_income_distribution_evolution.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_payback_period_trends(self, output_dir="results"):
        """
        FIXED: Plot payback period trends over time and by income class.
        """
        payback_data = self.metrics.get_payback_period_trends()
        
        if payback_data.empty:
            print("⚠️ plot_payback_period_trends: No payback data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Average payback by income class over time
        ax = axes[0, 0]
        for income_class in sorted(payback_data['IncomeClass'].unique()):
            class_data = payback_data[payback_data['IncomeClass'] == income_class]
            ax.plot(class_data['Year'], class_data['MeanPayback'], 
                label=f'Class {income_class}', linewidth=2, marker='o')
        
        ax.set_title("Average Payback Period by Income Class")
        ax.set_xlabel("Year")
        ax.set_ylabel("Payback Period (years)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)  # Reasonable payback period limit
        
        # Plot 2: Median payback by income class
        ax = axes[0, 1]
        for income_class in sorted(payback_data['IncomeClass'].unique()):
            class_data = payback_data[payback_data['IncomeClass'] == income_class]
            ax.plot(class_data['Year'], class_data['MedianPayback'], 
                label=f'Class {income_class}', linewidth=2, marker='s')
        
        ax.set_title("Median Payback Period by Income Class")
        ax.set_xlabel("Year")
        ax.set_ylabel("Payback Period (years)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 25)
        
        # Plot 3: Adoption rate vs payback period
        ax = axes[1, 0]
        final_year = payback_data['Year'].max()
        final_data = payback_data[payback_data['Year'] == final_year]
        
        ax.scatter(final_data['MeanPayback'], final_data['AdoptionRate'] * 100,
                s=100, alpha=0.7, c=final_data['IncomeClass'], cmap='RdYlBu_r')
        ax.set_title(f"Adoption Rate vs Payback Period (Year {final_year})")
        ax.set_xlabel("Average Payback Period (years)")
        ax.set_ylabel("Adoption Rate (%)")
        ax.grid(True, alpha=0.3)
        
        # Add colorbar for income classes
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Income Class')
        
        # Plot 4: Payback period distribution
        ax = axes[1, 1]
        all_payback = []
        for _, row in final_data.iterrows():
            # Approximate distribution based on mean and count
            all_payback.extend([row['MeanPayback']] * int(row['Count']))
        
        if all_payback:
            ax.hist(all_payback, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
            ax.set_title(f"Payback Period Distribution (Year {final_year})")
            ax.set_xlabel("Payback Period (years)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/payback_period_trends.png", dpi=300, bbox_inches="tight")
        plt.close()