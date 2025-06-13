"""
WEST Performance Optimization Visualization
==========================================

Generate plots showing polymer-enhanced configurations that outperform WEST tokamak.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# WEST baseline performance
WEST_BASELINE = {
    'confinement_time_s': 1337,
    'power_MW': 2.0
}

def create_optimization_visualizations():
    """Create comprehensive visualizations of WEST-beating configurations."""
    
    os.makedirs('west_optimization_results', exist_ok=True)
    
    # Optimization results from analysis
    configurations = [
        {'name': 'Enhanced HTS\nMaterials', 'confinement': 2485, 'power': 0.83, 'gain': 1.86},
        {'name': 'AI-Optimized\nCoil Geometry', 'confinement': 5650, 'power': 0.79, 'gain': 4.23},
        {'name': 'Liquid Metal\nDivertor', 'confinement': 3419, 'power': 1.52, 'gain': 2.56},
        {'name': 'Dynamic ELM\nMitigation', 'confinement': 2848, 'power': 1.66, 'gain': 2.13},
        {'name': 'Combined\nSynergistic', 'confinement': 11130, 'power': 0.56, 'gain': 8.32}
    ]
    
    # 1. Confinement Time Comparison
    plt.figure(figsize=(12, 8))
    
    names = [config['name'] for config in configurations]
    confinements = [config['confinement'] for config in configurations]
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'mediumorchid']
    
    bars = plt.bar(names, confinements, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add WEST baseline line
    plt.axhline(y=WEST_BASELINE['confinement_time_s'], color='red', linestyle='--', 
                linewidth=3, label=f'WEST Record: {WEST_BASELINE["confinement_time_s"]}s')
    
    # Add value labels on bars
    for bar, conf in zip(bars, confinements):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 200,
                f'{conf:.0f}s', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title('Polymer-Enhanced Fusion: Confinement Time Comparison\\nAll Configurations Beat WEST World Record', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Confinement Time (seconds)', fontsize=12, fontweight='bold')
    plt.xlabel('Optimization Configuration', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add improvement annotations
    for i, (bar, config) in enumerate(zip(bars, configurations)):
        improvement = config['confinement'] - WEST_BASELINE['confinement_time_s']
        plt.annotate(f'+{improvement:.0f}s\\n({config["gain"]:.1f}x)', 
                    xy=(bar.get_x() + bar.get_width()/2, config['confinement']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color='darkred', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('west_optimization_results/confinement_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Power Requirements Comparison
    plt.figure(figsize=(12, 8))
    
    powers = [config['power'] for config in configurations]
    
    bars = plt.bar(names, powers, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add WEST baseline line
    plt.axhline(y=WEST_BASELINE['power_MW'], color='red', linestyle='--', 
                linewidth=3, label=f'WEST Power: {WEST_BASELINE["power_MW"]}MW')
    
    # Add value labels on bars
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{power:.2f}MW', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.title('Polymer-Enhanced Fusion: Power Requirements\\nAll Configurations Require Less Power Than WEST', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Required Power (MW)', fontsize=12, fontweight='bold')
    plt.xlabel('Optimization Configuration', fontsize=12, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    # Add power savings annotations
    for i, (bar, config) in enumerate(zip(bars, configurations)):
        savings = WEST_BASELINE['power_MW'] - config['power']
        savings_pct = savings / WEST_BASELINE['power_MW'] * 100
        plt.annotate(f'-{savings:.2f}MW\\n(-{savings_pct:.1f}%)', 
                    xy=(bar.get_x() + bar.get_width()/2, config['power']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, color='darkgreen', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('west_optimization_results/power_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance Ratio Scatter Plot
    plt.figure(figsize=(12, 10))
    
    gains = [config['gain'] for config in configurations]
    power_ratios = [WEST_BASELINE['power_MW'] / config['power'] for config in configurations]
    performance_ratios = [g * pr for g, pr in zip(gains, power_ratios)]
    
    # Create scatter plot
    scatter = plt.scatter(gains, power_ratios, s=[200 * pr for pr in performance_ratios], 
                         c=colors, alpha=0.7, edgecolors='black', linewidths=2)
    
    # Add WEST baseline point
    plt.scatter([1], [1], s=300, c='red', marker='*', edgecolors='black', 
               linewidths=2, label='WEST Baseline', zorder=10)
    
    # Add configuration labels
    for i, config in enumerate(configurations):
        plt.annotate(config['name'], 
                    (gains[i], power_ratios[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.7))
    
    plt.xlabel('Confinement Time Gain (× WEST)', fontsize=12, fontweight='bold')
    plt.ylabel('Power Efficiency Gain (× WEST)', fontsize=12, fontweight='bold')
    plt.title('Polymer-Enhanced Fusion Performance\\nBubble Size = Overall Performance Ratio', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add quadrant labels
    plt.text(0.5, 3.5, 'Better Power\\nEfficiency Only', ha='center', va='center', 
             fontsize=10, style='italic', alpha=0.7,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.text(7, 0.8, 'Better Confinement\\nOnly', ha='center', va='center', 
             fontsize=10, style='italic', alpha=0.7,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    plt.text(7, 3.5, 'OPTIMAL\\nQUADRANT', ha='center', va='center', 
             fontsize=12, fontweight='bold', color='darkgreen',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('west_optimization_results/performance_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Combined Performance Timeline
    plt.figure(figsize=(14, 8))
    
    # Create timeline showing progression
    x_pos = np.arange(len(configurations))
    
    # Plot confinement times
    ax1 = plt.subplot(111)
    bars1 = ax1.bar(x_pos - 0.2, confinements, 0.4, label='Confinement Time (s)', 
                   color='skyblue', alpha=0.8, edgecolor='black')
    
    # Add WEST line for confinement
    ax1.axhline(y=WEST_BASELINE['confinement_time_s'], color='red', linestyle='--', 
                linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Optimization Configuration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Confinement Time (seconds)', fontsize=12, fontweight='bold', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    
    # Create second y-axis for power
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + 0.2, powers, 0.4, label='Required Power (MW)', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    # Add WEST line for power
    ax2.axhline(y=WEST_BASELINE['power_MW'], color='red', linestyle='--', 
                linewidth=2, alpha=0.7)
    
    ax2.set_ylabel('Required Power (MW)', fontsize=12, fontweight='bold', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    plt.title('Polymer-Enhanced Fusion: Simultaneous Improvement\\nHigher Confinement + Lower Power vs WEST', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Add WEST reference annotations
    ax1.text(0.5, WEST_BASELINE['confinement_time_s'] + 500, 'WEST Record', 
             fontsize=10, color='red', fontweight='bold')
    ax2.text(3.5, WEST_BASELINE['power_MW'] + 0.1, 'WEST Power', 
             fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('west_optimization_results/combined_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Optimization visualizations created:")
    print("  • confinement_comparison.png - Confinement time vs WEST")
    print("  • power_comparison.png - Power requirements vs WEST") 
    print("  • performance_scatter.png - Performance ratio analysis")
    print("  • combined_timeline.png - Simultaneous improvements")
    print("\\nAll plots saved to: west_optimization_results/")

if __name__ == "__main__":
    create_optimization_visualizations()
