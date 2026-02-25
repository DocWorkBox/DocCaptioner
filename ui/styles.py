from nicegui import ui

# --- Color Palette (Apple-inspired Light Mode) ---
COLOR_PRIMARY = '#007AFF' # iOS Blue
COLOR_BG = '#F2F2F7'      # System Gray 6
COLOR_CARD = '#FFFFFF'
COLOR_TEXT_PRIMARY = '#1C1C1E'
COLOR_TEXT_SECONDARY = '#8E8E93'
COLOR_BORDER = '#E5E5EA'

# --- Component Styles ---
# Buttons: Pill shape, bold text, smooth transition, Apple-style
BTN_BASE = 'rounded-full px-6 py-2 font-semibold transition-all duration-200 shadow-sm active:scale-95 text-sm tracking-wide'
BTN_PRIMARY = f'{BTN_BASE} !bg-blue-600 !text-white hover:!bg-blue-700 hover:shadow-md'
BTN_SECONDARY = f'{BTN_BASE} !bg-white !text-gray-700 border border-gray-200 hover:!bg-gray-50 hover:border-gray-300'
BTN_DANGER = f'{BTN_BASE} !bg-red-500 !text-white hover:!bg-red-600 hover:shadow-md'
BTN_GHOST = 'rounded-full px-4 py-2 !text-gray-600 hover:!bg-gray-100 transition-colors font-medium text-sm'

# Cards: Rounded corners (2xl), subtle shadow, clean white background
CARD_STYLE = 'rounded-2xl bg-white shadow-sm border border-gray-100 p-6'
# For inner cards or less prominent ones
CARD_INNER_STYLE = 'rounded-xl bg-gray-50 border border-gray-100 p-4'

# Inputs: Clean, rounded (xl), focus ring
# Note: NiceGUI/Quasar inputs are complex. We apply classes to the outer element.
# 'outlined' prop in Quasar already gives a border. We just round it more.
INPUT_STYLE = 'w-full rounded-xl' 
# We might need to override Quasar's internal border-radius via CSS or props.
# Quasar's 'rounded-borders' prop helps, but custom CSS is better for 'xl'.

# Typography
HEADER_STYLE = 'text-2xl font-bold text-gray-900 tracking-tight'
SUBHEADER_STYLE = 'text-lg font-semibold text-gray-700'
LABEL_STYLE = 'text-xs font-medium text-gray-500 uppercase tracking-wide'

# Global CSS Injection
GLOBAL_CSS = '''
<style>
    /* Font: Inter (Modern, Clean) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #F2F2F7; /* Apple System Gray 6 */
        color: #1C1C1E;
    }
    
    /* Custom Scrollbar (Thin & Modern) */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #C7C7CC;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #AEAEB2;
    }

    /* Apple-like Glass Effect for Header */
    .glass-header {
        background: rgba(255, 255, 255, 0.85);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
        position: sticky;
        top: 0;
        z-index: 50;
    }

    /* Quasar Overrides for Modern Look */
    .q-field__control {
        border-radius: 12px !important; /* Rounded Inputs */
    }
    .q-btn {
        border-radius: 9999px !important; /* Pill Buttons */
        text-transform: none !important; /* No ALL CAPS */
        font-weight: 600 !important;
    }
    .q-card {
        border-radius: 16px !important; /* Rounded Cards */
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06) !important; /* Soft Shadow */
    }
    
    /* Mobile Responsive Styles (Ported from layout.py) */
    @media (max-width: 768px) {
        /* 强制 Splitter 变为垂直布局 */
        .responsive-splitter {
            flex-direction: column !important;
            height: auto !important; /* 让高度自适应内容 */
            overflow-y: auto !important; /* 允许页面滚动 */
        }
        
        /* 面板宽度强制 100% */
        .responsive-splitter .q-splitter__panel {
            width: 100% !important;
            height: auto !important;
        }
        
        /* 上半部分（画廊）：固定高度，允许内部滚动 */
        .responsive-splitter .q-splitter__before {
            height: 50vh !important;
            border-bottom: 4px solid #e5e7eb;
        }
        
        /* 下半部分（功能区）：占满剩余空间或自适应 */
        .responsive-splitter .q-splitter__after {
            min-height: 50vh !important;
            height: auto !important;
        }
        
        /* 隐藏原有的分割条 */
        .responsive-splitter .q-splitter__separator {
            display: none !important;
        }
        
        /* 修复 Tabs 在移动端的显示 */
        .q-tabs__content {
            flex-wrap: wrap !important;
        }
    }
</style>
'''
