import re

def update_xai():
    with open('dashboard/src/components/AdvancedXAI.jsx', 'r', encoding='utf-8') as f:
        code = f.read()

    # 1. Imports & Helpers
    code = code.replace(
        "import React, { useState, useEffect, useCallback } from 'react';",
        "import React, { useState, useEffect, useCallback } from 'react';\nimport { Brain, Flame, BarChart3, FlaskConical, Dna, ScatterChart, Waves, LineChart, ListOrdered, Lightbulb, ChevronDown, Check, AlertTriangle, Play, Loader2 } from 'lucide-react';\n\nconst SpinStyle = () => (\n  <style>{`\n    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }\n  `}</style>\n);"
    )

    # 2. PlotCard loading/error
    code = code.replace(
        "<div style={{ fontSize: '2rem', marginBottom: 8 }}>⏳</div>",
        "<div style={{ display: 'flex', justifyContent: 'center', marginBottom: 12, color: '#06b6d4' }}><Loader2 size={32} style={{ animation: 'spin 2s linear infinite' }} /></div>"
    )
    code = code.replace(
        "Click <strong>\"▶ Run SHAP\"</strong> or <strong>\"▶ Run LIME\"</strong> above if nothing appears.",
        "Click <strong><Play size={12} fill=\"currentColor\" style={{verticalAlign:'-1px'}}/> Run SHAP</strong> or <strong><Play size={12} fill=\"currentColor\" style={{verticalAlign:'-1px'}}/> Run LIME</strong> above if nothing appears."
    )
    code = code.replace(
        "? '⚠ Chart not found. Click \"▶ Run SHAP\" or \"▶ Run LIME\" above to generate it.'",
        "? <><AlertTriangle size={16} style={{verticalAlign: '-3px', marginRight: 6}} /> Chart not found. Click <strong><Play size={12} fill=\"currentColor\" style={{verticalAlign:'-1px'}}/> Run SHAP</strong> or <strong><Play size={12} fill=\"currentColor\" style={{verticalAlign:'-1px'}}/> Run LIME</strong> above to generate it.</>"
    )
    code = code.replace(
        ": '⚠ No image path available.'}",
        ": <><AlertTriangle size={16} style={{verticalAlign: '-3px', marginRight: 6}} /> No image path available.</>}"
    )

    # 3. Tabs
    code = code.replace("{ id: 'attention',  label: '🔥 Attention Heatmap' }", "{ id: 'attention',  label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><Flame size={16} /> Attention Heatmap</span> }")
    code = code.replace("{ id: 'shap',       label: '📊 SHAP Analysis' }", "{ id: 'shap',       label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><BarChart3 size={16} /> SHAP Analysis</span> }")
    code = code.replace("{ id: 'lime',       label: '🧪 LIME Analysis' }", "{ id: 'lime',       label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><FlaskConical size={16} /> LIME Analysis</span> }")
    code = code.replace("{ id: 'hotspots',   label: '🧬 Mutation Hotspots' }", "{ id: 'hotspots',   label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><Dna size={16} /> Mutation Hotspots</span> }")

    # 4. Header
    code = code.replace("<span style={{ fontSize: '1.3rem' }}>🧠</span>", "<Brain size={24} color=\"#06b6d4\" />")
    code = code.replace(
        "<span style={{ fontSize: '1rem', transition: 'transform 0.3s', transform: open ? 'rotate(180deg)' : 'none', color: '#06b6d4' }}>▼</span>",
        "<span style={{ display: 'flex', alignItems: 'center', transition: 'transform 0.3s', transform: open ? 'rotate(180deg)' : 'none', color: '#06b6d4' }}><ChevronDown size={20} /></span>"
    )

    # 5. RunBtn
    code = code.replace(
        "setMsg(r.ok ? '✓ Done! Refresh to see new charts.' : `Error: ${d.detail}`);",
        "setMsg(r.ok ? <><Check size={14} style={{verticalAlign: '-2px'}} /> Done! Refresh to see new charts.</> : <><AlertTriangle size={14} style={{verticalAlign: '-2px'}} /> Error: {d.detail}</>);"
    )
    code = code.replace(
        "setMsg('⚠ Could not reach API.');",
        "setMsg(<><AlertTriangle size={14} style={{verticalAlign: '-2px'}} /> Could not reach API.</>);"
    )
    code = code.replace(
        "outline: '1px solid #06b6d444', fontWeight: 600, fontSize: '0.85rem',",
        "outline: '1px solid #06b6d444', fontWeight: 600, fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: 6,"
    )
    code = code.replace(
        "{running ? '⏳ Running…' : label}",
        "{running ? <Loader2 size={16} style={{ animation: 'spin 2s linear infinite' }} /> : <Play size={14} fill=\"currentColor\" />} {running ? 'Running…' : label.replace('▶ ', '')}"
    )
    code = code.replace("msg.startsWith('✓')", "msg.props?.children?.[0]?.type === Check")

    # 6. Section Titles
    code = code.replace('icon="🔵"', 'icon={<ScatterChart size={18} color="#0ea5e9" />}')
    code = code.replace('icon="📊"', 'icon={<BarChart3 size={18} color="#0ea5e9" />}')
    code = code.replace('icon="🌊"', 'icon={<Waves size={18} color="#0ea5e9" />}')
    code = code.replace('icon="📈"', 'icon={<LineChart size={18} color="#0ea5e9" />}')
    code = code.replace('icon="🔢"', 'icon={<ListOrdered size={18} color="#0ea5e9" />}')
    code = code.replace('icon="🔥"', 'icon={<Flame size={18} color="#f59e0b" />}')
    code = code.replace('icon="💡"', 'icon={<Lightbulb size={18} color="#eab308" />}')

    # Add SpinStyle inside main component
    code = code.replace(
        "<div style={{ marginTop: '2rem', borderRadius: 16, overflow: 'hidden', border: '1px solid rgba(148,163,184,0.18)', background: 'rgba(15,23,42,0.4)' }}>",
        "<div style={{ marginTop: '2rem', borderRadius: 16, overflow: 'hidden', border: '1px solid rgba(148,163,184,0.18)', background: 'rgba(15,23,42,0.4)' }}>\n      <SpinStyle />"
    )

    code = code.replace(
        "<div style={{ textAlign: 'center', color: '#475569', padding: '2rem' }}>⏳ Loading XAI data…</div>",
        "<div style={{ textAlign: 'center', color: '#475569', padding: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}><Loader2 size={20} style={{ animation: 'spin 2s linear infinite' }} /> Loading XAI data…</div>"
    )

    with open('dashboard/src/components/AdvancedXAI.jsx', 'w', encoding='utf-8') as f:
        f.write(code)


def update_analytics():
    with open('dashboard/src/components/ResearchAnalytics.jsx', 'r', encoding='utf-8') as f:
        code = f.read()

    code = code.replace(
        "import React, { useState, useEffect } from 'react';",
        "import React, { useState, useEffect } from 'react';\nimport { BarChart3, TrendingUp, Brain, Microscope, Flame, Dna, Hourglass, AlertTriangle, ChevronDown, Loader2 } from 'lucide-react';\n\nconst SpinStyle = () => (\n  <style>{`\n    @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }\n  `}</style>\n);"
    )

    # Loading
    code = code.replace(
        "<span style={{ fontSize: '1.5rem' }}>⏳</span>",
        "<Loader2 size={32} color=\"#06b6d4\" style={{ animation: 'spin 2s linear infinite' }} />"
    )

    # Tabs
    code = code.replace("{ id: 'confusion', label: '📊 Confusion Matrix' }", "{ id: 'confusion', label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><BarChart3 size={16} /> Confusion Matrix</span> }")
    code = code.replace("{ id: 'statistics', label: '📈 Statistical Analysis' }", "{ id: 'statistics', label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><TrendingUp size={16} /> Statistical Analysis</span> }")
    code = code.replace("{ id: 'xai', label: '🧠 Explainable AI' }", "{ id: 'xai', label: <span style={{display: 'flex', alignItems: 'center', gap: 6}}><Brain size={16} /> Explainable AI</span> }")

    # Header
    code = code.replace("<span style={{ fontSize: '1.3rem' }}>🔬</span>", "<Microscope size={24} color=\"#06b6d4\" />")
    code = code.replace(
        "<span style={{ fontSize: '1.1rem', transition: 'transform 0.3s', transform: open ? 'rotate(180deg)' : 'rotate(0deg)', color: '#06b6d4' }}>▼</span>",
        "<span style={{ display: 'flex', alignItems: 'center', transition: 'transform 0.3s', transform: open ? 'rotate(180deg)' : 'none', color: '#06b6d4' }}><ChevronDown size={20} /></span>"
    )

    # Section titles
    code = code.replace('icon="🔥"', 'icon={<Flame size={18} color="#f59e0b" />}')
    code = code.replace('icon="🧬"', 'icon={<Dna size={18} color="#0ea5e9" />}')
    code = code.replace('icon="🔬"', 'icon={<Microscope size={18} color="#8b5cf6" />}')

    # Errors
    code = code.replace("⚠ {error}", "<><AlertTriangle size={16} style={{verticalAlign: '-3px', marginRight: 6}} /> {error}</>")

    code = code.replace(
        "<div style={{ marginTop: '2.5rem', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16, overflow: 'hidden', background: 'rgba(15,23,42,0.4)' }}>",
        "<div style={{ marginTop: '2.5rem', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 16, overflow: 'hidden', background: 'rgba(15,23,42,0.4)' }}>\n      <SpinStyle />"
    )

    with open('dashboard/src/components/ResearchAnalytics.jsx', 'w', encoding='utf-8') as f:
        f.write(code)

if __name__ == '__main__':
    update_xai()
    update_analytics()
    print("Done replacing emojis with Lucide React icons.")
