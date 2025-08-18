# -*- coding: utf-8 -*-
from typing import List, Dict, Any
from markdown_it import MarkdownIt
from markdown_it.token import Token

def parse_markdown(md_text: str) -> List[Dict[str, Any]]:
    """
    Returns a list of block items:
    { type: 'heading'|'paragraph'|'list_item'|'code'|'table'|'link', 
      level?: int, text: str, line_start:int, line_end:int }
    """
    md = MarkdownIt()
    tokens: List[Token] = md.parse(md_text)
    blocks = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.type == 'heading_open':
            level = int(t.tag[1])
            inline = tokens[i+1] if i+1 < len(tokens) and tokens[i+1].type == 'inline' else None
            text = inline.content if inline else ''
            blocks.append({
                'type': 'heading',
                'level': level,
                'text': text,
                'line_start': (t.map[0] + 1) if t.map else None,
                'line_end': (t.map[1]) if t.map else None
            })
            i += 3  # skip heading_open, inline, heading_close
            continue
        if t.type == 'paragraph_open':
            inline = tokens[i+1] if i+1 < len(tokens) and tokens[i+1].type == 'inline' else None
            text = inline.content if inline else ''
            blocks.append({
                'type': 'paragraph',
                'text': text,
                'line_start': (t.map[0] + 1) if t.map else None,
                'line_end': (t.map[1]) if t.map else None
            })
            i += 3
            continue
        if t.type == 'bullet_list_open' or t.type == 'ordered_list_open':
            list_type = 'bullet' if t.type.startswith('bullet') else 'ordered'
            j = i+1
            while j < len(tokens) and tokens[j].type != ('bullet_list_close' if list_type=='bullet' else 'ordered_list_close'):
                if tokens[j].type == 'list_item_open':
                    inline = tokens[j+2] if j+2 < len(tokens) and tokens[j+2].type == 'inline' else None
                    text = inline.content if inline else ''
                    blocks.append({
                        'type': 'list_item',
                        'list_type': list_type,
                        'text': text,
                        'line_start': (tokens[j].map[0] + 1) if tokens[j].map else None,
                        'line_end': (tokens[j].map[1]) if tokens[j].map else None
                    })
                j += 1
            i = j + 1
            continue
        if t.type == 'fence':
            blocks.append({
                'type': 'code',
                'info': t.info,
                'text': t.content,
                'line_start': (t.map[0] + 1) if t.map else None,
                'line_end': (t.map[1]) if t.map else None
            })
            i += 1
            continue
        i += 1
    return blocks
