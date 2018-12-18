#!/usr/bin/env python3
# coding: utf8

# Author: Lenz Furrer, 2018


'''
Common tools for terminology resource processing.
'''


from collections import namedtuple


# Common format for terminology entries.
DictEntry = namedtuple('DictEntry', 'name id alt def_ syn')


class Terminology:
    '''
    Terminology indexed by names and IDs.
    '''
    def __init__(self, dict_entries, remove_ambiguous=False):
        self._by_name = {}
        self._by_id = {}
        self._index(dict_entries)
        if remove_ambiguous:
            self.remove_ambiguous()

    def _index(self, entries):
        for entry in entries:
            self.add(entry)

    def add(self, entry):
        '''
        Update the terminology with a DictEntry object.
        '''
        self._add(self._by_name, entry, entry.name, entry.syn)
        self._add(self._by_id, entry, entry.id, entry.alt)

    @staticmethod
    def _add(index, entry, main, secondary):
        for elem in (main, *secondary):
            index.setdefault(elem, []).append(entry)

    def remove_ambiguous(self):
        '''
        Remove ambiguous names from the terminology.
        '''
        delenda = []
        for name, entries in self._by_name.items():
            if len(entries) > 1:
                # There might be multiple entries with the same ID --
                # that doesn't count as ambiguity.
                if len(set(e.id for e in entries)) > 1:
                    # Don't delete right away --
                    # dict size shouldn't change during iteration.
                    delenda.append(name)
        for name in delenda:
            self._remove_ambiguous(name)

    def _remove_ambiguous(self, name):
        # Remove from self._by_name.
        entries = self._by_name.pop(name)
        # Remove from self._by_id.
        for entry in entries:
            self._remove_name_entry(entry, name)

    def _remove_name_entry(self, entry, name):
        newentry = entry._replace(syn=tuple(n for n in entry.syn if n != name))
        if newentry.name == name:
            try:
                newentry = newentry._replace(name=newentry.syn[0],
                                             syn=newentry.syn[1:])
            except IndexError:
                # No synonyms left. Delete the entry altogether.
                newentry = None
        for id_ in (entry.id, *entry.alt):
            entries = self._by_id[id_]
            try:
                entries.remove(entry)
            except ValueError:
                # It has already gone.
                continue
            if newentry is not None:
                entries.append(newentry)
            elif not entries:
                # If deleting this name means that this ID has no more names
                # associated with it, then the name is undeleted for this ID.
                # Re-insert the original entry here and in the name index.
                entries.append(entry)
                if entry not in self._by_name.setdefault(name, []):
                    self._by_name[name].append(entry)

    def has_id(self, id_):
        '''
        Is there an entry with this ID (canonical or alternative)?
        '''
        return id_ in self._by_id

    def has_name(self, name):
        '''
        Is there an entry mentioning this name?
        '''
        return name in self._by_name

    def ids(self, names):
        '''
        Get all (preferred) IDs associated with these names.
        '''
        return set(e.id for name in names for e in self._by_name.get(name, ()))

    def names(self, ids):
        '''
        Get all names and synonyms associated with these IDs.
        '''
        names = set()
        for id_ in ids:
            for e in self._by_id.get(id_, ()):
                names.add(e.name)
                names.update(e.syn)
        return names

    def definitions(self, name, include_ids=False):
        '''
        Get all definitions associated with this name.

        By default, return a set of definitions.
        If include_ids is True, return a dict instead,
        having definitions as keys and the respective IDs
        as values.
        '''
        entries = self._by_name.get(name, ())
        if not include_ids:
            return set(e.def_ for e in entries)
        else:
            defs = {}
            for e in entries:
                defs.setdefault(e.def_, set()).add(e.id)
            return defs

    def canonical_ids(self, id_):
        '''
        Get the preferred ID of all entries that list this ID as alternative.
        '''
        return set(e.id for e in self._by_id[id_])

    def iter_ids(self):
        '''
        Iterate over all IDs.
        '''
        yield from self._by_id

    def iter_names(self):
        '''
        Iterate over all names.
        '''
        yield from self._by_name
