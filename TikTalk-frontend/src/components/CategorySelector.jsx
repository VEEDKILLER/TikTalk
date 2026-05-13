const CATEGORY_CONFIG = {
  daily_life:     { emoji: '🏠', label: 'Daily Life',    desc: 'Home & family routines',   color: 'amber' },
  school:         { emoji: '🏫', label: 'School',         desc: 'Classroom & playground',   color: 'blue' },
  outdoor:        { emoji: '🌳', label: 'Outdoor',        desc: 'Parks, beaches & fields',  color: 'green' },
  community:      { emoji: '🏘️', label: 'Community',     desc: 'Markets & neighbourhood',  color: 'purple' },
  nature:         { emoji: '🌿', label: 'Nature',         desc: 'Gardens, zoos & forests',  color: 'emerald' },
  festivals:      { emoji: '🎉', label: 'Festivals',      desc: 'Celebrations & holidays',  color: 'rose' },
  helping:        { emoji: '🤝', label: 'Helping',        desc: 'Kindness & good deeds',    color: 'sky' },
  transportation: { emoji: '🚌', label: 'Transport',      desc: 'Buses, trains & planes',   color: 'orange' },
}

const CHARACTER_CONFIG = [
  { value: 'boy',                    emoji: '👦', label: 'Boy' },
  { value: 'girl',                   emoji: '👧', label: 'Girl' },
  { value: 'child',                  emoji: '🧒', label: 'Child' },
  { value: 'mother',                 emoji: '👩', label: 'Mother' },
  { value: 'father',                 emoji: '👨', label: 'Father' },
  { value: 'grandmother',            emoji: '👵', label: 'Grandma' },
  { value: 'grandfather',            emoji: '👴', label: 'Grandpa' },
  { value: 'young boy',              emoji: '🧒', label: 'Young Boy' },
  { value: 'young girl',             emoji: '👧', label: 'Young Girl' },
  { value: 'brother and sister',     emoji: '👫', label: 'Siblings' },
  { value: 'group of children',      emoji: '👨‍👩‍👧‍👦', label: 'Group' },
  { value: 'little girl with pigtails', emoji: '👧', label: 'Pigtail Girl' },
  { value: 'boy wearing glasses',    emoji: '🤓', label: 'Boy w/ Glasses' },
  { value: 'girl in a school uniform', emoji: '👩‍🎓', label: 'School Uniform' },
]

const COLOR_MAP = {
  amber:   { ring: 'ring-amber-400',   bg: 'bg-amber-50',   badge: 'bg-amber-100 text-amber-700',   icon: 'bg-amber-100' },
  blue:    { ring: 'ring-blue-400',    bg: 'bg-blue-50',    badge: 'bg-blue-100 text-blue-700',     icon: 'bg-blue-100' },
  green:   { ring: 'ring-green-400',   bg: 'bg-green-50',   badge: 'bg-green-100 text-green-700',   icon: 'bg-green-100' },
  purple:  { ring: 'ring-purple-400',  bg: 'bg-purple-50',  badge: 'bg-purple-100 text-purple-700', icon: 'bg-purple-100' },
  emerald: { ring: 'ring-emerald-400', bg: 'bg-emerald-50', badge: 'bg-emerald-100 text-emerald-700', icon: 'bg-emerald-100' },
  rose:    { ring: 'ring-rose-400',    bg: 'bg-rose-50',    badge: 'bg-rose-100 text-rose-700',     icon: 'bg-rose-100' },
  sky:     { ring: 'ring-sky-400',     bg: 'bg-sky-50',     badge: 'bg-sky-100 text-sky-700',       icon: 'bg-sky-100' },
  orange:  { ring: 'ring-orange-400',  bg: 'bg-orange-50',  badge: 'bg-orange-100 text-orange-700', icon: 'bg-orange-100' },
}

export default function CategorySelector({
  categories, characters,
  selected, character,
  onCategory, onCharacter,
}) {
  const selectedConfig = CATEGORY_CONFIG[selected]
  const selectedColor = selectedConfig ? COLOR_MAP[selectedConfig.color] : null

  return (
    <div className="space-y-6">

      {/* ── Scene grid ────────────────────────────────────────────────────── */}
      <div>
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
          1 · Pick a scene
        </p>
        <div className="grid grid-cols-4 gap-2.5">
          {categories.map((cat) => {
            const cfg = CATEGORY_CONFIG[cat] || { emoji: '🖼️', label: cat, desc: '', color: 'blue' }
            const clr = COLOR_MAP[cfg.color]
            const isSelected = selected === cat
            return (
              <button
                key={cat}
                onClick={() => onCategory(cat)}
                className={`
                  group relative flex flex-col items-center gap-2 p-3 rounded-2xl border-2
                  transition-all duration-200 text-left
                  ${isSelected
                    ? `border-transparent ring-2 ${clr.ring} ${clr.bg} shadow-md scale-105`
                    : 'border-gray-100 bg-white hover:border-gray-200 hover:shadow-sm hover:scale-102'
                  }
                `}
              >
                {/* Emoji bubble */}
                <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-2xl
                  transition-colors ${isSelected ? clr.icon : 'bg-gray-50 group-hover:bg-gray-100'}`}>
                  {cfg.emoji}
                </div>
                <div className="text-center">
                  <p className="text-xs font-semibold text-gray-700 leading-tight">{cfg.label}</p>
                  <p className="text-[10px] text-gray-400 mt-0.5 leading-tight">{cfg.desc}</p>
                </div>
                {isSelected && (
                  <span className={`absolute top-1.5 right-1.5 w-2 h-2 rounded-full ${clr.badge.split(' ')[0].replace('bg-', 'bg-').replace('100', '400')}`} />
                )}
              </button>
            )
          })}
        </div>

        {/* Selected category pill */}
        {selectedConfig && selectedColor && (
          <div className={`mt-3 inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium ${selectedColor.badge}`}>
            <span>{selectedConfig.emoji}</span>
            <span>{selectedConfig.label} selected</span>
            <span className="opacity-60">— {selectedConfig.desc}</span>
          </div>
        )}
      </div>

      {/* ── Character picker ──────────────────────────────────────────────── */}
      <div>
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-3">
          2 · Pick a character
        </p>
        <div className="flex flex-wrap gap-2">
          {CHARACTER_CONFIG
            .filter(c => characters.includes(c.value))
            .map((c) => {
              const isSelected = character === c.value
              return (
                <button
                  key={c.value}
                  onClick={() => onCharacter(c.value)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl border-2 text-xs font-medium
                    transition-all duration-150
                    ${isSelected
                      ? 'border-blue-400 bg-blue-50 text-blue-700 shadow-sm scale-105'
                      : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300 hover:bg-gray-50'
                    }`}
                >
                  <span className="text-base leading-none">{c.emoji}</span>
                  <span>{c.label}</span>
                </button>
              )
            })}
        </div>
      </div>

    </div>
  )
}
