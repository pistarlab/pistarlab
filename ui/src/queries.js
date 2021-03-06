
import gql from "graphql-tag";

const GET_RECENT_AGENTS = gql`
{
  recentAgents {
    id
    ident
    name
    created
    specId
    spec{
      id
      ident
      displayedName
      extensionId

    }
    lastCheckpoint
    tags{
        edges{
            node{
                id
                tagId
            }
        }
    }
    recentSessions {
      id
      ident
      status
      envSpecId
      sessionType
      parentSessionId
      envSpec{
        id
          ident
          environment {
              id
              ident
          }
          
        }
      
      
    }
  }
}
`;

const GET_RECENT_AGENT_SMALL = gql`
{
  recentAgents {
    id
    ident
    name
    created
    specId
    spec{
        id
        ident
        displayedName
    }
  }
}
`;



const GET_ALL_SESSIONS = gql `
  query GetSessions($first: Int, $after: String, $before:String, $archived:Boolean) {
    sessions: allSessions(
      sort: CREATED_DESC
      first: $first
      before: $before
      after: $after,
      filters: {
            archived:$archived  
        }
    ) {
      pageInfo {
        startCursor
        endCursor
        hasNextPage
        hasPreviousPage
      }
      edges {
        node {
          id
          ident
          created
          envSpecId
          envSpec{
            id
            displayedName
          }          
          label
          comments
          status
          config
          sessionType
          agentId
          agent{
            id
            ident
            specId
            name
          }
            task {
                id
              ident
        }
        
          summary 
        }
      }
    }
  }
`;

const GET_ALL_ENVS=  gql `
query {
  environments: allEnvironments {
    edges {
      node {
          id
          ident
          description
          displayedName
          extensionId
          categories
          collection
          extensionVersion
          version
          disabled
          specs {
              id
              ident
              description
              displayedName
              tags
              
              envType
              entryPoint
              meta
          }
      }
    }
  }
}
`


export {GET_ALL_ENVS, GET_RECENT_AGENTS,GET_RECENT_AGENT_SMALL,GET_ALL_SESSIONS };
