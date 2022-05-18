" Vim syntax file
" Language:     Garnet
" Maintainer:   Simon Heath <icefox@dreamquest.io>
" Last Change:  Jul 04, 2020

" Currently just copied from the rust syntax highlighting with
" some changes.
" To use it, put it in ~/.config/nvim/syntax
" and run
"
" :set syntax=garnet
"
" You can make vim autodetect it by adding the following to your
" ~/.config/nvim/init.vim:
"
" au BufRead,BufNewFile *.gt		set filetype=garnet

if version < 600
	syntax clear
elseif exists("b:current_syntax")
	finish
endif

" Syntax definitions {{{1
" Basic keywords {{{2
syn keyword   garnetConditional match if else then elseif
syn keyword   garnetRepeat for loop while do
syn keyword   garnetTypedef type nextgroup=garnetIdentifier skipwhite skipempty
syn keyword   garnetStructure struct enum nextgroup=garnetIdentifier skipwhite skipempty
syn keyword   garnetUnion union nextgroup=garnetIdentifier skipwhite skipempty contained
syn match garnetUnionContextual /\<union\_s\+\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*/ transparent contains=garnetUnion
syn keyword   garnetBinop    as and or not xor band bor bnot bxor

syn match     garnetAssert      "\<assert\(\w\)*!" contained
syn match     garnetPanic       "\<panic\(\w\)*!" contained
syn keyword   garnetKeyword     end
syn keyword   garnetKeyword     break
syn keyword   garnetKeyword     box nextgroup=garnetBoxPlacement skipwhite skipempty
syn keyword   garnetKeyword     continue
syn keyword   garnetKeyword     extern nextgroup=garnetExternCrate,garnetObsoleteExternMod skipwhite skipempty
syn keyword   garnetKeyword     fn nextgroup=garnetFuncName skipwhite skipempty
syn keyword   garnetKeyword     in impl let
syn keyword   garnetKeyword     pub nextgroup=garnetPubScope skipwhite skipempty
syn keyword   garnetKeyword     return
syn keyword   garnetSuper       super
syn keyword   garnetKeyword     unsafe where
syn keyword   garnetKeyword     use nextgroup=garnetModPath skipwhite skipempty
" FIXME: Scoped impl's name is also fallen in this category
syn keyword   garnetKeyword     mod trait nextgroup=garnetIdentifier skipwhite skipempty
syn keyword   garnetStorage     move uniq ref static const
syn match garnetDefault /\<default\ze\_s\+\(impl\|fn\|type\|const\)\>/

syn keyword   garnetInvalidBareKeyword crate

syn keyword garnetPubScopeCrate crate contained
syn match garnetPubScopeDelim /[()]/ contained
syn match garnetPubScope /([^()]*)/ contained contains=garnetPubScopeDelim,garnetPubScopeCrate,garnetSuper,garnetModPath,garnetModPathSep,garnetSelf transparent

syn keyword   garnetExternCrate crate contained nextgroup=garnetIdentifier,garnetExternCrateString skipwhite skipempty
" This is to get the `bar` part of `extern crate "foo" as bar;` highlighting.
syn match   garnetExternCrateString /".*"\_s*as/ contained nextgroup=garnetIdentifier skipwhite transparent skipempty contains=garnetString,garnetOperator
syn keyword   garnetObsoleteExternMod mod contained nextgroup=garnetIdentifier skipwhite skipempty

syn match     garnetIdentifier  contains=garnetIdentifierPrime "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained
syn match     garnetFuncName    "\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*" display contained

syn region    garnetBoxPlacement matchgroup=garnetBoxPlacementParens start="(" end=")" contains=TOP contained
" Ideally we'd have syntax rules set up to match arbitrary expressions. Since
" we don't, we'll just define temporary contained rules to handle balancing
" delimiters.
syn region    garnetBoxPlacementBalance start="(" end=")" containedin=garnetBoxPlacement transparent
syn region    garnetBoxPlacementBalance start="\[" end="\]" containedin=garnetBoxPlacement transparent
" {} are handled by garnetFoldBraces

syn region garnetMacroRepeat matchgroup=garnetMacroRepeatDelimiters start="$(" end=")" contains=TOP nextgroup=garnetMacroRepeatCount
syn match garnetMacroRepeatCount ".\?[*+]" contained
syn match garnetMacroVariable "$\w\+"

" Reserved (but not yet used) keywords {{{2
syn keyword   garnetReservedKeyword alignof become offsetof priv pure sizeof typeof unsized yield abstract virtual final override macro

" Built-in types {{{2
syn keyword   garnetType        isize usize Char Bool U8 U16 U32 U64 U128 F32
syn keyword   garnetType        F64 I8 I16 I32 I64 I128 str Self

" Things from the libstd v1 prelude (src/libstd/prelude/v1.rs) {{{2
" This section is just straight transformation of the contents of the prelude,
" to make it easy to update.

" Reexported core operators {{{3
syn keyword   garnetTrait       Copy Send Sized Sync
syn keyword   garnetTrait       Drop Fn FnMut FnOnce

" Reexported functions {{{3
" There’s no point in highlighting these; when one writes drop( or drop::< it
" gets the same highlighting anyway, and if someone writes `let drop = …;` we
" don’t really want *that* drop to be highlighted.
"syn keyword garnetFunction drop

" Reexported types and traits {{{3
syn keyword garnetTrait Box
syn keyword garnetTrait ToOwned
syn keyword garnetTrait Clone
syn keyword garnetTrait PartialEq PartialOrd Eq Ord
syn keyword garnetTrait AsRef AsMut Into From
syn keyword garnetTrait Default
syn keyword garnetTrait Iterator Extend IntoIterator
syn keyword garnetTrait DoubleEndedIterator ExactSizeIterator
syn keyword garnetEnum Option
syn keyword garnetEnumVariant Some None
syn keyword garnetEnum Result
syn keyword garnetEnumVariant Ok Err
syn keyword garnetTrait SliceConcatExt
syn keyword garnetTrait String ToString
syn keyword garnetTrait Vec

" Other syntax {{{2
syn keyword   garnetSelf        self
syn keyword   garnetBoolean     true false

" If foo::bar changes to foo.bar, change this ("::" to "\.").
" If foo::bar changes to Foo::bar, change this (first "\w" to "\u").
syn match     garnetModPath     "\w\(\w\)*::[^<]"he=e-3,me=e-3
syn match     garnetModPathSep  "::"

syn match     garnetFuncCall    "\w\(\w\)*("he=e-1,me=e-1
syn match     garnetFuncCall    "\w\(\w\)*::<"he=e-3,me=e-3 " foo::<T>();

" This is merely a convention; note also the use of [A-Z], restricting it to
" latin identifiers rather than the full Unicode uppercase. I have not used
" [:upper:] as it depends upon 'noignorecase'
"syn match     garnetCapsIdent    display "[A-Z]\w\(\w\)*"

syn match     garnetOperator     display "\%(+\|-\|/\|*\|=\|\^\|&\||\|!\|>\|<\|%\)=\?"
" This one isn't *quite* right, as we could have binary-& with a reference
syn match     garnetSigil        display /&\s\+[&~@*][^)= \t\r\n]/he=e-1,me=e-1
syn match     garnetSigil        display /[&~@*][^)= \t\r\n]/he=e-1,me=e-1
" This isn't actually correct; a closure with no arguments can be `|| { }`.
" Last, because the & in && isn't a sigil
" This is garnetArrowCharacter rather than garnetArrow for the sake of matchparen,
" so it skips the ->; see http://stackoverflow.com/a/30309949 for details.
syn match     garnetArrowCharacter display "->"
syn match     garnetQuestionMark display "?\([a-zA-Z]\+\)\@!"

syn match     garnetMacro       '\w\(\w\)*!' contains=garnetAssert,garnetPanic
syn match     garnetMacro       '#\w\(\w\)*' contains=garnetAssert,garnetPanic

syn match     garnetEscapeError   display contained /\\./
syn match     garnetEscape        display contained /\\\([nrt0\\'"]\|x\x\{2}\)/
syn match     garnetEscapeUnicode display contained /\\u{\x\{1,6}}/
syn match     garnetStringContinuation display contained /\\\n\s*/
syn region    garnetString      start=+b"+ skip=+\\\\\|\\"+ end=+"+ contains=garnetEscape,garnetEscapeError,garnetStringContinuation
syn region    garnetString      start=+"+ skip=+\\\\\|\\"+ end=+"+ contains=garnetEscape,garnetEscapeUnicode,garnetEscapeError,garnetStringContinuation,@Spell
syn region    garnetString      start='b\?r\z(#*\)"' end='"\z1' contains=@Spell

syn region    garnetAttribute   start="#!\?\[" end="\]" contains=garnetString,garnetDerive,garnetCommentLine,garnetCommentBlock,garnetCommentLineDocError,garnetCommentBlockDocError
syn region    garnetDerive      start="derive(" end=")" contained contains=garnetDeriveTrait
" This list comes from src/libsyntax/ext/deriving/mod.rs
" Some are deprecated (Encodable, Decodable) or to be removed after a new snapshot (Show).
syn keyword   garnetDeriveTrait contained Clone Hash garnetcEncodable garnetcDecodable Encodable Decodable PartialEq Eq PartialOrd Ord Rand Show Debug Default FromPrimitive Send Sync Copy

" Number literals
syn match     garnetDecNumber   display "\<[0-9][0-9_]*\%([iu]\%(size\|8\|16\|32\|64\|128\)\)\="
syn match     garnetHexNumber   display "\<0x[a-fA-F0-9_]\+\%([iu]\%(size\|8\|16\|32\|64\|128\)\)\="
syn match     garnetOctNumber   display "\<0o[0-7_]\+\%([iu]\%(size\|8\|16\|32\|64\|128\)\)\="
syn match     garnetBinNumber   display "\<0b[01_]\+\%([iu]\%(size\|8\|16\|32\|64\|128\)\)\="

" Special case for numbers of the form "1." which are float literals, unless followed by
" an identifier, which makes them integer literals with a method call or field access,
" or by another ".", which makes them integer literals followed by the ".." token.
" (This must go first so the others take precedence.)
syn match     garnetFloat       display "\<[0-9][0-9_]*\.\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\|\.\)\@!"
" To mark a number as a normal float, it must have at least one of the three things integral values don't have:
" a decimal point and more numbers; an exponent; and a type suffix.
syn match     garnetFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\%([eE][+-]\=[0-9_]\+\)\=\(f32\|f64\)\="
syn match     garnetFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\=\%([eE][+-]\=[0-9_]\+\)\(f32\|f64\)\="
syn match     garnetFloat       display "\<[0-9][0-9_]*\%(\.[0-9][0-9_]*\)\=\%([eE][+-]\=[0-9_]\+\)\=\(f32\|f64\)"

" For the benefit of delimitMate
syn region garnetLifetimeCandidate display start=/&'\%(\([^'\\]\|\\\(['nrt0\\\"]\|x\x\{2}\|u{\x\{1,6}}\)\)'\)\@!/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=garnetSigil,garnetLifetime
syn region garnetGenericRegion display start=/<\%('\|[^[cntrl:][:space:][:punct:]]\)\@=')\S\@=/ end=/>/ contains=garnetGenericLifetimeCandidate
syn region garnetGenericLifetimeCandidate display start=/\%(<\|,\s*\)\@<='/ end=/[[:cntrl:][:space:][:punct:]]\@=\|$/ contains=garnetSigil,garnetLifetime

"garnetLifetime must appear before garnetCharacter, or chars will get the lifetime highlighting
syn match     garnetLifetime    display "\'\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*"
syn match     garnetLabel       display "\'\%([^[:cntrl:][:space:][:punct:][:digit:]]\|_\)\%([^[:cntrl:][:punct:][:space:]]\|_\)*:"
syn match   garnetCharacterInvalid   display contained /b\?'\zs[\n\r\t']\ze'/
" The groups negated here add up to 0-255 but nothing else (they do not seem to go beyond ASCII).
syn match   garnetCharacterInvalidUnicode   display contained /b'\zs[^[:cntrl:][:graph:][:alnum:][:space:]]\ze'/
syn match   garnetCharacter   /b'\([^\\]\|\\\(.\|x\x\{2}\)\)'/ contains=garnetEscape,garnetEscapeError,garnetCharacterInvalid,garnetCharacterInvalidUnicode
syn match   garnetCharacter   /'\([^\\]\|\\\(.\|x\x\{2}\|u{\x\{1,6}}\)\)'/ contains=garnetEscape,garnetEscapeUnicode,garnetEscapeError,garnetCharacterInvalid

syn match garnetShebang /\%^#![^[].*/
syn region garnetCommentLine                                                  start="--"                      end="$"   contains=garnetTodo,@Spell
syn region garnetCommentLineDoc                                               start="--\%(-\|!\)"         end="$"   contains=garnetTodo,@Spell
syn region garnetCommentLineDocError                                          start="--\%(//\@!\|!\)"         end="$"   contains=garnetTodo,@Spell contained
syn region garnetCommentBlock             matchgroup=garnetCommentBlock         start="/\-\%(!\|\-[*/]\@!\)\@!" end="\-/" contains=garnetTodo,garnetCommentBlockNest,@Spell
syn region garnetCommentBlockDoc          matchgroup=garnetCommentBlockDoc      start="/\-\%(!\|\-[*/]\@!\)"    end="\-/" contains=garnetTodo,garnetCommentBlockDocNest,@Spell
syn region garnetCommentBlockDocError     matchgroup=garnetCommentBlockDocError start="/\-\%(!\|\-[*/]\@!\)"    end="\-/" contains=garnetTodo,garnetCommentBlockDocNestError,@Spell contained
syn region garnetCommentBlockNest         matchgroup=garnetCommentBlock         start="/\-"                     end="\-/" contains=garnetTodo,garnetCommentBlockNest,@Spell contained transparent
syn region garnetCommentBlockDocNest      matchgroup=garnetCommentBlockDoc      start="/\-"                     end="\-/" contains=garnetTodo,garnetCommentBlockDocNest,@Spell contained transparent
syn region garnetCommentBlockDocNestError matchgroup=garnetCommentBlockDocError start="/\-"                     end="\-/" contains=garnetTodo,garnetCommentBlockDocNestError,@Spell contained transparent
" FIXME: this is a really ugly and not fully correct implementation. Most
" importantly, a case like ``/* */*`` should have the final ``*`` not being in
" a comment, but in practice at present it leaves comments open two levels
" deep. But as long as you stay away from that particular case, I *believe*
" the highlighting is correct. Due to the way Vim's syntax engine works
" (greedy for start matches, unlike garnet's tokeniser which is searching for
" the earliest-starting match, start or end), I believe this cannot be solved.
" Oh you who would fix it, don't bother with things like duplicating the Block
" rules and putting ``\*\@<!`` at the start of them; it makes it worse, as
" then you must deal with cases like ``/*/**/*/``. And don't try making it
" worse with ``\%(/\@<!\*\)\@<!``, either...

syn keyword garnetTodo contained TODO FIXME XXX NB NOTE

" Folding rules {{{2
" Trivial folding rules to begin with.
" FIXME: use the AST to make really good folding
syn region garnetFoldBraces start="{" end="}" transparent fold

" Default highlighting {{{1
hi def link garnetDecNumber       garnetNumber
hi def link garnetHexNumber       garnetNumber
hi def link garnetOctNumber       garnetNumber
hi def link garnetBinNumber       garnetNumber
hi def link garnetIdentifierPrime garnetIdentifier
hi def link garnetTrait           garnetType
hi def link garnetDeriveTrait     garnetTrait

hi def link garnetMacroRepeatCount   garnetMacroRepeatDelimiters
hi def link garnetMacroRepeatDelimiters   Macro
hi def link garnetMacroVariable Define
hi def link garnetSigil         StorageClass
hi def link garnetEscape        Special
hi def link garnetEscapeUnicode garnetEscape
hi def link garnetEscapeError   Error
hi def link garnetStringContinuation Special
hi def link garnetString        String
hi def link garnetCharacterInvalid Error
hi def link garnetCharacterInvalidUnicode garnetCharacterInvalid
hi def link garnetCharacter     Character
hi def link garnetNumber        Number
hi def link garnetBoolean       Boolean
hi def link garnetEnum          garnetType
hi def link garnetEnumVariant   garnetConstant
hi def link garnetConstant      Constant
hi def link garnetSelf          Constant
hi def link garnetFloat         Float
hi def link garnetArrowCharacter garnetOperator
hi def link garnetOperator      Operator
hi def link garnetKeyword       Keyword
hi def link garnetBinop         StorageClass
hi def link garnetTypedef       Keyword " More precise is Typedef, but it doesn't feel right for garnet
hi def link garnetStructure     Keyword " More precise is Structure
hi def link garnetUnion         garnetStructure
hi def link garnetPubScopeDelim Delimiter
hi def link garnetPubScopeCrate garnetKeyword
hi def link garnetSuper         garnetKeyword
hi def link garnetReservedKeyword Error
hi def link garnetRepeat        Conditional
hi def link garnetConditional   Conditional
hi def link garnetIdentifier    Identifier
hi def link garnetCapsIdent     garnetIdentifier
hi def link garnetModPath       Include
hi def link garnetModPathSep    Delimiter
hi def link garnetFunction      Function
hi def link garnetFuncName      Function
hi def link garnetFuncCall      Function
hi def link garnetShebang       Comment
hi def link garnetCommentLine   Comment
hi def link garnetCommentLineDoc SpecialComment
hi def link garnetCommentLineDocError Error
hi def link garnetCommentBlock  garnetCommentLine
hi def link garnetCommentBlockDoc garnetCommentLineDoc
hi def link garnetCommentBlockDocError Error
hi def link garnetAssert        PreCondit
hi def link garnetPanic         PreCondit
hi def link garnetMacro         Macro
hi def link garnetType          Type
hi def link garnetTodo          Todo
hi def link garnetAttribute     PreProc
hi def link garnetDerive        PreProc
hi def link garnetDefault       StorageClass
hi def link garnetStorage       StorageClass
hi def link garnetObsoleteStorage Error
hi def link garnetLifetime      Special
hi def link garnetLabel         Label
hi def link garnetInvalidBareKeyword Error
hi def link garnetExternCrate   garnetKeyword
hi def link garnetObsoleteExternMod Error
hi def link garnetBoxPlacementParens Delimiter
hi def link garnetQuestionMark  Special

" Other Suggestions:
" hi garnetAttribute ctermfg=cyan
" hi garnetDerive ctermfg=cyan
" hi garnetAssert ctermfg=yellow
" hi garnetPanic ctermfg=red
" hi garnetMacro ctermfg=magenta

syn sync minlines=200
syn sync maxlines=500

let b:current_syntax = "garnet"
