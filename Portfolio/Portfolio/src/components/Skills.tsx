import { BarChart3, Wrench, TrendingUp } from "lucide-react";

const Skills = () => {
  const skillCategories = [
    {
      icon: BarChart3,
      category: "Data Analysis & Reporting",
      skills: [
        "Writing SQL queries to extract, aggregate, and analyze business data",
        "Identifying trends, risks, and performance gaps through structured analysis",
        "Building business-focused reports and dashboards to support decision-making",

      ],
      gradient: "from-primary to-primary/70",
    },
    {
      icon: Wrench,
      category: "Tools & Technologies",
      skills: [
        "SQL",
        "Python (pandas)",
        "Excel (Pivot Tables, Lookups)",
        "Power BI (Dashboards, Data Modeling)",
      ],
      gradient: "from-secondary to-secondary/70",
    },
    {
      icon: TrendingUp,
      category: "Analytics Capabilities",
      skills: [
      "Data cleaning and preparation for accurate analysis",
      "Exploratory Data Analysis (EDA) to uncover patterns and anomalies",
      "KPI tracking and performance analysis for business monitoring",
      "Customer and operational analytics to support optimization decisions",
      ],
      gradient: "from-primary/80 to-secondary/80",
    },
  ];

  return (
    <section id="skills" className="py-20 relative overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-b from-background via-primary/5 to-background" />
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16 animate-fade-in">
            <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold mb-4 bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Skills & Expertise
            </h2>
            <div className="w-20 h-1 bg-gradient-to-r from-primary to-secondary mx-auto mb-8 rounded-full" />
            <p className="text-base sm:text-lg text-muted-foreground max-w-2xl mx-auto">
              A comprehensive toolkit for transforming data into actionable business insights
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
            {skillCategories.map((category, categoryIndex) => {
              const Icon = category.icon;
              return (
                <div
                  key={category.category}
                  className="glass-card p-6 sm:p-8 hover:shadow-xl transition-all duration-300 animate-slide-up group hover:scale-105"
                  style={{ animationDelay: `${categoryIndex * 0.1}s` }}
                >
                  <div className="flex items-center gap-4 mb-6">
                    <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${category.gradient} flex items-center justify-center group-hover:scale-110 transition-transform`}>
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <h3 className="text-lg sm:text-xl font-bold text-foreground">
                      {category.category}
                    </h3>
                  </div>

                  <ul className="space-y-3">
                    {category.skills.map((skill, index) => (
                      <li key={index} className="flex items-start gap-3">
                        <span className="text-primary mt-1 text-sm">▪</span>
                        <span className="text-sm sm:text-base text-muted-foreground leading-relaxed">
                          {skill}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
};

export default Skills;
