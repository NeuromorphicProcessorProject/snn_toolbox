# -*- coding: utf-8 -*-
"""
This is a sub-function of importAedat. 
This function processes the headers of an Aedat file. 
(as well as any attached prefs files). 
The .aedat file format is documented here:
http://inilabs.com/support/software/fileformat/
    
2015_12_11 Work in progress: 
    - Reading from a separate prefs file is not implemented yet.  
    - It would be neater (more readable) to turn the xml cell array into a
      structure.
"""

from snntoolbox.datasets.aedat.ImportAedatBasicSourceName import \
    import_aedat_basic_source_name


def import_aedat_headers(info):
    # Start from the beginning of the file (should not be necessary)
    """

    Parameters
    ----------
    info :

    Returns
    -------

    """
    info['fileHandle'].seek(0)

    # From version 3.1 there is an unambiguous division between header and data:
    # A line like this: '#!END-HEADER\r\n'
    # However, for all older files there's no guarantee that the first character
    # in the data would not be '#'. We ignore this - we look in the next 
    # unread position and if it is not # we exit. 

    info['xml'] = {}

    # Assume the format version is 1 unless a header of the version number is
    # found
    info['formatVersion'] = 1

    # Read the first character
    is_comment = '#' in str(info['fileHandle'].read(1))
    while is_comment:

        # Read the rest of the line
        line = info['fileHandle'].readline().decode('utf-8')
        # File format
        if line[: 8] == '!AER-DAT':
            info['formatVersion'] = int(line[8: -4])

        # Pick out the source
        # Version 2.0 encodes it like this:
        if line[: 9] == ' AEChip: ':
            # Ignore everything the class path and only use what follows the
            # final dot
            start_prefix = line.rfind('.')
            info['sourceFromFile'] = import_aedat_basic_source_name(
                line[start_prefix + 1:])
        # Version 3.0 encodes it like this
        # The following ignores any trace of previous sources
        # (prefixed with a minus sign)
        if line[: 8] == ' Source ':
            start_prefix = line.find(':')  # There should be only one colon
            if 'sourceFromFile' in info:
                pass
                # One source has already been added; convert to a cell array if
                # it has not already been done

                # NOT HANDLED YET:            

                # if ~iscell(info.sourceFromFile)
                #    info.sourceFromFile = {info.sourceFromFile};
                # info.sourceFromFile = [info.sourceFromFile line[start_prefix
                #  + 2 : ];
            else:
                info['sourceFromFile'] = line[start_prefix + 2:]

        # Pick out date and time of recording

        # Version 2.0 encodes it like this:
        # # created Thu Dec 03 14:47:00 CET 2015
        if line[: 9] == ' created ':
            info['dateTime'] = line[9:]

        # Version 3.0 encodes it like this:
        # # Start-Time: #Y-#m-#d #H:#M:#S (TZ#z)\r\n
        if line[: 13] == ' Start-Time: ':
            info['dateTime'] = line[13:]

        """# Parse xml, adding it to output as a cell array, in a field
        called 'xml'. # This is done by maintaining a cell array which is
        inside out as it is # constructed - as a level of the hierarchy is
        descended, everything is # pushed down into the first position of a
        cell array, and as the # hierarchy is ascended, the first node is
        popped back up and the nodes # that have been added to the right are
        pushed down inside it.
        
        # If <node> then descend hierarchy - do this by taking the existing
        # cell array and putting it into another cell array
        if strncmp(line, '<node', 5)
            nameOfNode = line(length('<node name="') + 1 : end - length('">'));
            info.xml = {info.xml nameOfNode};

        # </node> - ascend hierarchy - take everything to the right of the
        # initial cell array and put it inside the inital cell array
        elseif strncmp(line, '</node>', 7)
            parent = info.xml{1};
            child = info.xml(2:end);
            info.xml = [parent {child}];

          # <entry> - Add a field to the struct
            elseif strncmp(line, '<entry ', 7)
            # Find the division between key and value
            endOfKey = strfind(line, '" value="');
            key = line(length('<entry key="') + 1 : endOfKey - 1);
            value = line(endOfKey + length('" value="') : end - length('"/>'));
        info.xml{end + 1} = {key value};
        end
        # Gets the next line, including line ending chars
         line = native2unicode(fgets(info.fileHandle)); 
        """
        # Read ahead the first character of the next line to complete the
        # while loop
        is_comment = '#' in str(info['fileHandle'].read(1))

    # We have read ahead one byte looking for '#', and not found it.
    # Now wind back one to be in the right place to start reading
    info['fileHandle'].seek(-1, 1)
    info['beginningOfDataPointer'] = info['fileHandle'].tell()

    """
    # If a device is specified in input, does it match the derived source?
    if isfield(info, 'source')
        info.source = importAedat_basicSourceName(info.source);
        if isfield(info, 'sourceFromFile') && ~strcmp(info.source,
         info.sourceFromFile)
            fprintf('The source given as input, "#s", doesn''t match the source
            declared in the file, "#s"; assuming the source given as input.\n',
             inputSource, info.Source);
        end
    elseif ~isfield(info, 'sourceFromFile')
    # If no source was detected, assume it was from a DVS128	
    info.source = 'Dvs128';
    else
        info.source = info.sourceFromFile;
    end
    if isfield(info, 'sourceFromFile') 
        # Clean up
        rmfield(info, 'sourceFromFile');
    end
        
    """

    return info
